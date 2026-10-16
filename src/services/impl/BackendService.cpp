#include "services/impl/BackendService.h"
#include "inference/ImagePipeline.h"

#include <triton/core/tritonserver.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <exception>
#include <mutex>
#include <set>
#include <thread>
#include <utility>

namespace triton_backend
{
namespace
{

using Json = nlohmann::json;
using Clock = std::chrono::steady_clock;

double milliseconds(Clock::time_point start)
{
    return std::chrono::duration<double, std::milli>(Clock::now() - start).count();
}

void discardError(TRITONSERVER_Error* error) noexcept
{
    if (error != nullptr) TRITONSERVER_ErrorDelete(error);
}

void check(TRITONSERVER_Error* error, const char* operation)
{
    if (error == nullptr) return;
    const std::unique_ptr<TRITONSERVER_Error, decltype(&TRITONSERVER_ErrorDelete)> owned(error, TRITONSERVER_ErrorDelete);
    const std::string message = std::string(operation) + ": " + TRITONSERVER_ErrorMessage(error);
    throw std::runtime_error(message);
}

struct ServerDeleter
{
    void operator()(TRITONSERVER_Server* value) const noexcept
    { discardError(TRITONSERVER_ServerDelete(value)); }
};
struct OptionsDeleter
{
    void operator()(TRITONSERVER_ServerOptions* value) const noexcept
    { discardError(TRITONSERVER_ServerOptionsDelete(value)); }
};
struct AllocatorDeleter
{
    void operator()(TRITONSERVER_ResponseAllocator* value) const noexcept
    { discardError(TRITONSERVER_ResponseAllocatorDelete(value)); }
};
struct MessageDeleter
{
    void operator()(TRITONSERVER_Message* value) const noexcept
    { discardError(TRITONSERVER_MessageDelete(value)); }
};
struct RequestDeleter
{
    void operator()(TRITONSERVER_InferenceRequest* value) const noexcept
    { discardError(TRITONSERVER_InferenceRequestDelete(value)); }
};
struct ResponseDeleter
{
    void operator()(TRITONSERVER_InferenceResponse* value) const noexcept
    { discardError(TRITONSERVER_InferenceResponseDelete(value)); }
};

Json readMessage(TRITONSERVER_Message* raw)
{
    std::unique_ptr<TRITONSERVER_Message, MessageDeleter> message(raw);
    const char* data = nullptr;
    std::size_t length = 0;
    check(TRITONSERVER_MessageSerializeToJson(message.get(), &data, &length), "读取 Triton 信息");
    return Json::parse(data, data + length);
}

TRITONSERVER_Error* allocateOutput(
    TRITONSERVER_ResponseAllocator*, const char*, std::size_t byteSize,
    TRITONSERVER_MemoryType, std::int64_t, void*, void** buffer, void** bufferUser,
    TRITONSERVER_MemoryType* actualType, std::int64_t* actualId
) noexcept
{
    *buffer = byteSize == 0 ? nullptr : std::malloc(byteSize);
    *bufferUser = nullptr;
    *actualType = TRITONSERVER_MEMORY_CPU;
    *actualId = 0;
    if (byteSize != 0 && *buffer == nullptr)
    {
        return TRITONSERVER_ErrorNew(TRITONSERVER_ERROR_INTERNAL, "无法分配 Triton 输出缓冲区");
    }
    return nullptr;
}

TRITONSERVER_Error* releaseOutput(
    TRITONSERVER_ResponseAllocator*, void* buffer, void*, std::size_t,
    TRITONSERVER_MemoryType, std::int64_t
) noexcept
{
    std::free(buffer);
    return nullptr;
}

struct RequestState
{
    std::vector<float> input;
    std::vector<float> output;
    std::string requestId;
    std::string modelName;
    std::int64_t modelVersion{};
    std::mutex mutex;
    std::condition_variable completed;
    bool requestReleased{};
    bool responseFinal{};
    std::exception_ptr error;
};

using StateHandle = std::shared_ptr<RequestState>;

void requestReleased(TRITONSERVER_InferenceRequest* request, std::uint32_t flags, void* user) noexcept
{
    if ((flags & TRITONSERVER_REQUEST_RELEASE_ALL) == 0) return;
    std::unique_ptr<StateHandle> handle(static_cast<StateHandle*>(user));
    const StateHandle state = *handle;
    discardError(TRITONSERVER_InferenceRequestDelete(request));
    {
        std::lock_guard<std::mutex> lock(state->mutex);
        state->requestReleased = true;
    }
    state->completed.notify_all();
}

void responseCompleted(TRITONSERVER_InferenceResponse* raw, std::uint32_t flags, void* user) noexcept
{
    auto* handle = static_cast<StateHandle*>(user);
    const StateHandle state = *handle;
    std::unique_ptr<TRITONSERVER_InferenceResponse, ResponseDeleter> response(raw);
    std::vector<float> output;
    std::exception_ptr error;
    try
    {
        if (response == nullptr)
        {
            throw std::runtime_error("Triton 未返回模型输出");
        }
        if (TRITONSERVER_Error* responseError = TRITONSERVER_InferenceResponseError(response.get()))
        {
            // ResponseError 由 response 持有，不能单独 ErrorDelete。
            throw std::runtime_error(TRITONSERVER_ErrorMessage(responseError));
        }
        const char* responseId = nullptr;
        const char* modelName = nullptr;
        std::int64_t modelVersion = 0;
        check(TRITONSERVER_InferenceResponseId(response.get(), &responseId), "读取响应 ID");
        check(TRITONSERVER_InferenceResponseModel(response.get(), &modelName, &modelVersion), "读取响应模型");
        if (responseId == nullptr || responseId != state->requestId || modelName == nullptr ||
            modelName != state->modelName || modelVersion != state->modelVersion)
        {
            throw std::runtime_error("Triton 响应的请求 ID 或模型版本不匹配");
        }
        std::uint32_t count = 0;
        check(TRITONSERVER_InferenceResponseOutputCount(response.get(), &count), "读取输出数量");
        if (count != 1) throw std::runtime_error("模型必须只返回 output0");
        const char* name = nullptr;
        TRITONSERVER_DataType type{};
        const std::int64_t* shape = nullptr;
        std::uint64_t dimensions = 0;
        const void* buffer = nullptr;
        std::size_t bytes = 0;
        TRITONSERVER_MemoryType memoryType{};
        std::int64_t memoryId = 0;
        void* bufferUser = nullptr;
        check(TRITONSERVER_InferenceResponseOutput(response.get(), 0, &name, &type, &shape,
              &dimensions, &buffer, &bytes, &memoryType, &memoryId, &bufferUser), "读取 output0");
        if (name == nullptr || std::strcmp(name, "output0") != 0 || type != TRITONSERVER_TYPE_FP32 ||
            dimensions != 3 || shape == nullptr || shape[0] != 1 || shape[1] != 300 || shape[2] != 6 ||
            bytes != 300U * 6U * sizeof(float) || buffer == nullptr ||
            memoryType != TRITONSERVER_MEMORY_CPU)
        {
            throw std::runtime_error("模型输出必须为 CPU FP32 output0 [1,300,6]");
        }
        output.resize(300 * 6);
        std::memcpy(output.data(), buffer, bytes);
        if ((flags & TRITONSERVER_RESPONSE_COMPLETE_FINAL) == 0)
        {
            throw std::runtime_error("当前接口不支持多响应模型");
        }
    }
    catch (...)
    {
        error = std::current_exception();
    }
    response.reset();
    {
        std::lock_guard<std::mutex> lock(state->mutex);
        if (error) state->error = std::move(error);
        else state->output = std::move(output);
        if ((flags & TRITONSERVER_RESPONSE_COMPLETE_FINAL) != 0) state->responseFinal = true;
    }
    state->completed.notify_all();
    if ((flags & TRITONSERVER_RESPONSE_COMPLETE_FINAL) != 0) delete handle;
}

void validateModels(const RuntimeConfig& runtime, const std::vector<ModelConfig>& models)
{
    if (runtime.maxInFlight == 0 || runtime.stopTimeoutSeconds <= 0 ||
        runtime.root.empty() || runtime.repository.empty() || models.empty())
    {
        throw std::invalid_argument("Triton 路径、模型、在途上限或停止超时无效");
    }
    validateModelCatalog(models);
    if (!std::filesystem::is_directory(runtime.root) || !std::filesystem::is_directory(runtime.repository))
    {
        throw std::invalid_argument("Triton 安装目录或模型仓库不存在");
    }
}

} // namespace

struct BackendService::Impl
{
    RuntimeConfig runtime;
    const std::vector<ModelConfig> models;
    std::unique_ptr<TRITONSERVER_ResponseAllocator, AllocatorDeleter> allocator;
    std::unique_ptr<TRITONSERVER_Server, ServerDeleter> server;
    mutable std::mutex mutex;
    std::mutex lifecycle;
    std::condition_variable drained;
    std::string state{"stopped"};
    std::string lastError;
    std::size_t inFlight{};
    struct Generation
    {
        std::size_t modelIndex{};
        std::uint64_t number{};
        std::string name;
        std::filesystem::path repository;
        Json metadata;
        std::size_t inFlight{};
        bool registered{};
        bool unloading{};
        std::string cleanupError;
        Clock::time_point retryAfter{};

        ~Generation()
        {
            if (!repository.empty())
            {
                std::error_code error;
                std::filesystem::remove_all(repository, error);
            }
        }
    };
    std::shared_ptr<Generation> activeGeneration;
    std::vector<std::shared_ptr<Generation>> retiring;
    std::optional<std::size_t> pendingModel;
    std::uint64_t nextGeneration{};
    bool stopping{};
    bool finishReaper{};
    std::thread reaper;
    Json serverMetadata;
    std::uint32_t apiMajor{};
    std::uint32_t apiMinor{};

    Impl(RuntimeConfig settings, std::vector<ModelConfig> configuredModels)
        : runtime(std::move(settings)), models(std::move(configuredModels))
    {
        validateModels(runtime, models);
        check(TRITONSERVER_ApiVersion(&apiMajor, &apiMinor), "查询 Triton API 版本");
        if (apiMajor != TRITONSERVER_API_VERSION_MAJOR || apiMinor < TRITONSERVER_API_VERSION_MINOR)
        {
            throw std::runtime_error("Triton 动态库与编译头文件 API 版本不兼容");
        }
        TRITONSERVER_ServerOptions* rawOptions = nullptr;
        check(TRITONSERVER_ServerOptionsNew(&rawOptions), "创建 Triton options");
        std::unique_ptr<TRITONSERVER_ServerOptions, OptionsDeleter> options(rawOptions);
        check(TRITONSERVER_ServerOptionsSetModelRepositoryPath(options.get(), runtime.repository.c_str()), "设置模型仓库");
        const auto backends = std::filesystem::path(runtime.root) / "backends";
        check(TRITONSERVER_ServerOptionsSetBackendDirectory(options.get(), backends.c_str()), "设置 backend 目录");
        check(TRITONSERVER_ServerOptionsSetModelControlMode(options.get(), TRITONSERVER_MODEL_CONTROL_EXPLICIT), "设置显式模型控制");
        check(TRITONSERVER_ServerOptionsSetExitOnError(options.get(), false), "设置加载错误处理");
        check(TRITONSERVER_ServerOptionsSetModelLoadRetryCount(options.get(), 0), "禁用模型加载重试");
        check(TRITONSERVER_ServerOptionsSetStrictModelConfig(options.get(), true), "设置严格模型配置");
        check(TRITONSERVER_ServerOptionsSetExitTimeout(options.get(), runtime.stopTimeoutSeconds), "设置退出超时");
        TRITONSERVER_Server* rawServer = nullptr;
        TRITONSERVER_Error* creationError = TRITONSERVER_ServerNew(&rawServer, options.get());
        server.reset(rawServer);
        check(creationError, "创建 Triton core");
        TRITONSERVER_ResponseAllocator* rawAllocator = nullptr;
        check(TRITONSERVER_ResponseAllocatorNew(&rawAllocator, allocateOutput, releaseOutput, nullptr), "创建输出 allocator");
        allocator.reset(rawAllocator);
        TRITONSERVER_Message* message = nullptr;
        check(TRITONSERVER_ServerMetadata(server.get(), &message), "查询 Triton core 信息");
        serverMetadata = readMessage(message);
        reaper = std::thread([this] { retireLoop(); });
    }

    bool modelReady(const Generation& generation) const
    {
        bool ready = false;
        const auto& model = models[generation.modelIndex];
        TRITONSERVER_Error* error = TRITONSERVER_ServerModelIsReady(server.get(), generation.name.c_str(), model.tritonVersion, &ready);
        if (error != nullptr)
        {
            discardError(error);
            return false;
        }
        return ready;
    }

    bool available(const ModelConfig& model) const
    {
        std::error_code error;
        return std::filesystem::is_regular_file(std::filesystem::path(runtime.repository) /
            model.tritonName / std::to_string(model.tritonVersion) / "model.plan", error);
    }

    Json describe(std::size_t index) const
    {
        const auto& model = models[index];
        const bool active = activeGeneration && activeGeneration->modelIndex == index && !stopping;
        Json description{{"id", model.id}, {"version", model.externalVersion}, {"tritonName", model.tritonName},
                    {"tritonVersion", model.tritonVersion}, {"inputSize", model.inputSize},
                    {"classCount", model.classNames.size()}, {"available", available(model)},
                    {"active", active}, {"ready", active && modelReady(*activeGeneration)},
                    {"batchSize", 1}, {"backend", "tensorrt"},
                    {"metadataSource", active ? "loaded_model" : "configuration"}};
        description["batching"] = Json{{"maxBatchSize", model.batching.maxBatchSize},
            {"preferredBatchSizes", model.batching.preferredBatchSizes},
            {"maxQueueDelayMicroseconds", model.batching.maxQueueDelayMicroseconds}};
        description["instanceCount"] = model.instanceCount;
        const std::int64_t batchDimension = model.batching.maxBatchSize > 0 ? -1 : 1;
        description["inputs"] = Json::array({Json{{"name", "images"}, {"datatype", "FP32"},
            {"shape", {batchDimension, 3, model.inputSize, model.inputSize}}}});
        description["outputs"] = Json::array({Json{{"name", "output0"}, {"datatype", "FP32"}, {"shape", {batchDimension, 300, 6}}}});
        if (active) description["generation"] = activeGeneration->number;
        return description;
    }

    Json selected() const { return activeGeneration ? describe(activeGeneration->modelIndex) : Json(nullptr); }

    void load(Generation& generation)
    {
        const auto& model = models[generation.modelIndex];
        std::string pattern = (std::filesystem::temp_directory_path() / "triton-back-runtime-XXXXXX").string();
        std::vector<char> path(pattern.begin(), pattern.end());
        path.push_back('\0');
        const auto directory = ::mkdtemp(path.data());
        if (directory == nullptr) throw std::runtime_error("无法创建候选模型临时仓库");
        generation.repository = directory;
        std::filesystem::create_directory_symlink(
            std::filesystem::absolute(std::filesystem::path(runtime.repository) / model.tritonName),
            generation.repository / "source");
        std::unique_ptr<TRITONSERVER_Parameter, decltype(&TRITONSERVER_ParameterDelete)> mapping(
            TRITONSERVER_ParameterNew("source", TRITONSERVER_PARAMETER_STRING, generation.name.c_str()),
            TRITONSERVER_ParameterDelete);
        if (!mapping) throw std::runtime_error("无法创建候选模型名称映射");
        const TRITONSERVER_Parameter* mappings[]{mapping.get()};
        check(TRITONSERVER_ServerRegisterModelRepository(server.get(), generation.repository.c_str(), mappings, 1), "注册候选模型仓库");
        generation.registered = true;
        // 每代独立名称隔离同名版本，加载失败时不会替换活动模型。
        std::vector<std::int64_t> inputDimensions{3, model.inputSize, model.inputSize};
        std::vector<std::int64_t> outputDimensions{300, 6};
        if (model.batching.maxBatchSize == 0)
        {
            inputDimensions.insert(inputDimensions.begin(), 1);
            outputDimensions.insert(outputDimensions.begin(), 1);
        }
        auto configurationDocument = Json{
            {"name", generation.name}, {"backend", "tensorrt"}, {"default_model_filename", "model.plan"},
            {"max_batch_size", model.batching.maxBatchSize}, {"version_policy", {{"specific", {{"versions", {model.tritonVersion}}}}}},
            {"input", Json::array({Json{{"name", "images"}, {"data_type", "TYPE_FP32"},
                {"dims", inputDimensions}}})},
            {"output", Json::array({Json{{"name", "output0"}, {"data_type", "TYPE_FP32"}, {"dims", outputDimensions}}})},
            {"instance_group", Json::array({Json{{"count", model.instanceCount}, {"kind", "KIND_GPU"}, {"gpus", {0}}}})}
        };
        if (model.batching.maxBatchSize > 0)
        {
            configurationDocument["dynamic_batching"] = Json{
                {"max_queue_delay_microseconds", model.batching.maxQueueDelayMicroseconds}};
            if (!model.batching.preferredBatchSizes.empty())
                configurationDocument["dynamic_batching"]["preferred_batch_size"] = model.batching.preferredBatchSizes;
        }
        const auto configuration = configurationDocument.dump();
        std::unique_ptr<TRITONSERVER_Parameter, decltype(&TRITONSERVER_ParameterDelete)> parameter(
            TRITONSERVER_ParameterNew("config", TRITONSERVER_PARAMETER_STRING, configuration.c_str()),
            TRITONSERVER_ParameterDelete);
        if (!parameter) throw std::runtime_error("无法创建模型加载参数");
        const TRITONSERVER_Parameter* parameters[]{parameter.get()};
        check(TRITONSERVER_ServerLoadModelWithParameters(server.get(), generation.name.c_str(), parameters, 1), "加载候选模型版本");
    }

    void cleanup(Generation& generation)
    {
        if (!generation.registered) return;
        check(TRITONSERVER_ServerUnloadModel(server.get(), generation.name.c_str()), "卸载退休模型");
        const auto deadline = Clock::now() + std::chrono::seconds(runtime.stopTimeoutSeconds);
        while (true)
        {
            TRITONSERVER_Message* message = nullptr;
            check(TRITONSERVER_ServerModelIndex(server.get(), 0, &message), "检查退休模型卸载状态");
            const auto index = readMessage(message);
            const bool busy = std::any_of(index.begin(), index.end(), [&](const auto& entry)
            {
                const auto state = entry.value("state", "");
                return entry.value("name", "") == generation.name &&
                    (state == "READY" || state == "LOADING" || state == "UNLOADING");
            });
            if (!busy) break;
            if (Clock::now() >= deadline) throw std::runtime_error("退休模型卸载超时");
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
        check(TRITONSERVER_ServerUnregisterModelRepository(server.get(), generation.repository.c_str()), "注销退休模型仓库");
        generation.registered = false;
    }

    void activate(std::size_t index)
    {
        const auto& target = models[index];
        if (!available(target))
            throw InvalidInferenceRequest("MODEL_FILE_NOT_FOUND", "指定版本的 model.plan 不存在：" + target.id + "/" + target.externalVersion);
        const auto candidate = std::make_shared<Generation>();
        candidate->modelIndex = index;
        {
            std::lock_guard<std::mutex> lock(mutex);
            if (activeGeneration && activeGeneration->modelIndex == index && modelReady(*activeGeneration)) return;
            if (stopping) throw BackendNotReady("服务进程正在退出");
            candidate->number = ++nextGeneration;
            candidate->name = "triton_back_generation_" + std::to_string(candidate->number);
            pendingModel = index;
            state = activeGeneration ? "running" : "starting";
            lastError.clear();
        }
        try
        {
            load(*candidate);
            candidate->metadata = inspectModel(*candidate);
            const auto output = inferTensor(std::vector<float>(3U * target.inputSize * target.inputSize, 0.0F),
                                           *candidate, "prepare-" + std::to_string(candidate->number));
            for (const auto& detection : decodeDetections(output, LetterboxInfo{1, 0, 0, target.inputSize, target.inputSize}, 0.25F))
                if (target.classNames.count(detection.classId) == 0)
                    throw std::runtime_error("候选模型预热输出了未登记的类别");
            std::lock_guard<std::mutex> lock(mutex);
            // 准备成功后才切流；已接收任务继续持有旧代，退休线程按代计数清理。
            if (activeGeneration) retiring.push_back(activeGeneration);
            activeGeneration = candidate;
            pendingModel.reset();
            state = "running";
            drained.notify_all();
        }
        catch (const std::exception& error)
        {
            {
                std::lock_guard<std::mutex> lock(mutex);
                retiring.push_back(candidate);
                pendingModel.reset();
                state = activeGeneration ? "running" : "failed";
                lastError = error.what();
                drained.notify_all();
            }
            throw;
        }
    }

    Json inspectModel(const Generation& generation)
    {
        const auto& model = models[generation.modelIndex];
        if (!modelReady(generation)) throw std::runtime_error("模型加载后仍未就绪：" + model.id);
        TRITONSERVER_Message* message = nullptr;
        check(TRITONSERVER_ServerModelMetadata(server.get(), generation.name.c_str(), model.tritonVersion, &message), "查询模型信息");
        Json metadata = readMessage(message);
        const Json inputs = metadata.at("inputs");
        const Json outputs = metadata.at("outputs");
        const std::int64_t batchDimension = model.batching.maxBatchSize > 0 ? -1 : 1;
        if (inputs.size() != 1 || outputs.size() != 1 ||
            inputs[0].at("name") != "images" || inputs[0].at("datatype") != "FP32" ||
            inputs[0].at("shape") != Json::array({batchDimension, 3, model.inputSize, model.inputSize}) ||
            outputs[0].at("name") != "output0" || outputs[0].at("datatype") != "FP32" ||
            outputs[0].at("shape") != Json::array({batchDimension, 300, 6}))
        {
            throw std::runtime_error("模型输入输出不符合配置的 FP32 images / output0 契约：" + model.id);
        }
        std::uint32_t transaction = 0;
        void* properties = nullptr;
        check(TRITONSERVER_ServerModelTransactionProperties(server.get(), generation.name.c_str(), model.tritonVersion,
              &transaction, &properties), "查询模型响应策略");
        if ((transaction & TRITONSERVER_TXN_ONE_TO_ONE) == 0 || (transaction & TRITONSERVER_TXN_DECOUPLED) != 0)
        {
            throw std::runtime_error("当前服务只支持单请求单响应模型");
        }
        return metadata;
    }

    std::vector<float> inferTensor(std::vector<float> input, const Generation& generation, const std::string& requestId)
    {
        const auto& model = models[generation.modelIndex];
        auto state = std::make_shared<RequestState>();
        state->input = std::move(input);
        state->requestId = requestId;
        state->modelName = generation.name;
        state->modelVersion = model.tritonVersion;
        TRITONSERVER_InferenceRequest* raw = nullptr;
        check(TRITONSERVER_InferenceRequestNew(&raw, server.get(), generation.name.c_str(), model.tritonVersion), "创建推理请求");
        std::unique_ptr<TRITONSERVER_InferenceRequest, RequestDeleter> request(raw);
        check(TRITONSERVER_InferenceRequestSetId(request.get(), requestId.c_str()), "设置请求 ID");
        const std::int64_t shape[]{1, 3, model.inputSize, model.inputSize};
        check(TRITONSERVER_InferenceRequestAddInput(request.get(), "images", TRITONSERVER_TYPE_FP32, shape, 4), "设置输入 shape");
        check(TRITONSERVER_InferenceRequestAppendInputData(request.get(), "images", state->input.data(),
              state->input.size() * sizeof(float), TRITONSERVER_MEMORY_CPU, 0), "绑定输入数据");
        check(TRITONSERVER_InferenceRequestAddRequestedOutput(request.get(), "output0"), "设置请求输出");
        auto releaseHandle = std::make_unique<StateHandle>(state);
        auto responseHandle = std::make_unique<StateHandle>(state);
        check(TRITONSERVER_InferenceRequestSetReleaseCallback(request.get(), requestReleased, releaseHandle.get()), "设置请求释放回调");
        check(TRITONSERVER_InferenceRequestSetResponseCallback(request.get(), allocator.get(), nullptr,
              responseCompleted, responseHandle.get()), "设置响应回调");
        check(TRITONSERVER_ServerInferAsync(server.get(), request.get(), nullptr), "提交推理");
        request.release();
        releaseHandle.release();
        responseHandle.release();
        // 两个回调可按任意顺序到达；等待二者后，输入及 allocator 状态才可安全释放。
        std::unique_lock<std::mutex> lock(state->mutex);
        state->completed.wait(lock, [&state] { return state->requestReleased && state->responseFinal; });
        if (state->error) std::rethrow_exception(state->error);
        return std::move(state->output);
    }

    void retireLoop()
    {
        std::unique_lock<std::mutex> lock(mutex);
        while (!finishReaper)
        {
            auto selected = retiring.end();
            auto wakeAt = Clock::time_point::max();
            for (auto item = retiring.begin(); item != retiring.end(); ++item)
            {
                if ((*item)->inFlight != 0 || (*item)->unloading) continue;
                if ((*item)->retryAfter <= Clock::now()) { selected = item; break; }
                wakeAt = std::min(wakeAt, (*item)->retryAfter);
            }
            if (selected == retiring.end())
            {
                if (wakeAt == Clock::time_point::max()) drained.wait(lock);
                else drained.wait_until(lock, wakeAt);
                continue;
            }
            const auto generation = *selected;
            generation->unloading = true;
            lock.unlock();
            std::string error;
            try { cleanup(*generation); }
            catch (const std::exception& failure) { error = failure.what(); }
            lock.lock();
            if (error.empty())
                retiring.erase(std::find(retiring.begin(), retiring.end(), generation));
            else
            {
                generation->cleanupError = std::move(error);
                generation->retryAfter = Clock::now() + std::chrono::seconds(1);
                generation->unloading = false;
            }
            drained.notify_all();
        }
    }

    void releaseAdmission(const std::shared_ptr<Generation>& generation)
    {
        {
            std::lock_guard<std::mutex> lock(mutex);
            --inFlight;
            --generation->inFlight;
        }
        drained.notify_all();
    }
};

BackendService::BackendService(RuntimeConfig runtime, std::vector<ModelConfig> models)
    : impl_(std::make_unique<Impl>(std::move(runtime), std::move(models)))
{
}

BackendService::~BackendService()
{
    try { stop(); } catch (...) {}
    {
        std::unique_lock<std::mutex> lock(impl_->mutex);
        impl_->stopping = true;
        impl_->state = "stopping";
        impl_->drained.wait(lock, [this] { return impl_->inFlight == 0; });
        impl_->finishReaper = true;
        impl_->drained.notify_all();
    }
    if (impl_->reaper.joinable()) impl_->reaper.join();
    discardError(TRITONSERVER_ServerStop(impl_->server.get()));
    impl_->server.reset();
    impl_->allocator.reset();
    impl_->activeGeneration.reset();
    impl_->retiring.clear();
}

void BackendService::selectModel(const ModelSelection& selection)
{
    std::lock_guard<std::mutex> management(impl_->lifecycle);
    bool knownModel = false;
    for (std::size_t index = 0; index < impl_->models.size(); ++index)
    {
        const auto& model = impl_->models[index];
        if (model.id != selection.modelId) continue;
        knownModel = true;
        if (model.externalVersion != selection.version) continue;
        impl_->activate(index);
        return;
    }
    throw InvalidInferenceRequest(knownModel ? "MODEL_VERSION_NOT_FOUND" : "MODEL_NOT_FOUND",
        "未注册指定模型或版本：" + selection.modelId + "/" + selection.version);
}

void BackendService::stop()
{
    std::lock_guard<std::mutex> management(impl_->lifecycle);
    {
        std::unique_lock<std::mutex> lock(impl_->mutex);
        if (impl_->state == "stopped") return;
        impl_->stopping = true;
        impl_->state = "stopping";
        if (!impl_->drained.wait_for(lock, std::chrono::seconds(impl_->runtime.stopTimeoutSeconds),
                                    [this] { return impl_->inFlight == 0; }))
        {
            impl_->lastError = "停止超时：仍有推理请求正在排空";
            throw std::runtime_error(impl_->lastError);
        }
    }
    std::unique_lock<std::mutex> lock(impl_->mutex);
    if (impl_->activeGeneration) impl_->retiring.push_back(std::move(impl_->activeGeneration));
    impl_->drained.notify_all();
    if (!impl_->drained.wait_for(lock, std::chrono::seconds(impl_->runtime.stopTimeoutSeconds),
                               [this] { return impl_->retiring.empty(); }))
    {
        impl_->lastError = "进程退出时清理模型超时";
        throw std::runtime_error(impl_->lastError);
    }
    impl_->state = "stopped";
    impl_->stopping = false;
    impl_->lastError.clear();
}

bool BackendService::isReady() const
{
    std::lock_guard<std::mutex> lock(impl_->mutex);
    return !impl_->stopping && impl_->activeGeneration && impl_->modelReady(*impl_->activeGeneration);
}

InferenceResult BackendService::infer(
    const InferenceRequest& request, const std::string& modelId,
    const std::string& externalVersion, const std::string& requestId
)
{
    const auto started = Clock::now();
    const auto first = std::find_if(impl_->models.begin(), impl_->models.end(),
        [&modelId](const ModelConfig& model) { return model.id == modelId; });
    if (first == impl_->models.end()) throw InvalidInferenceRequest("MODEL_MISMATCH", "未配置请求的模型");
    const auto found = std::find_if(impl_->models.begin(), impl_->models.end(),
        [&](const ModelConfig& model) { return model.id == modelId && model.externalVersion == externalVersion; });
    if (found == impl_->models.end()) throw InvalidInferenceRequest("MODEL_VERSION_MISMATCH", "未配置请求的模型版本");
    if (!request.image && !request.pngImage)
        throw InvalidInferenceRequest("INVALID_RAW_IMAGE", "推理接口需要原始像素或 PNG 文件");
    if (request.image && request.pngImage)
        throw InvalidInferenceRequest("INVALID_IMAGE", "每个请求只能提供一种图像输入");
    if (!std::isfinite(request.confidenceThreshold) || request.confidenceThreshold < 0 || request.confidenceThreshold > 1)
    {
        throw InvalidInferenceRequest("INVALID_CONFIDENCE_THRESHOLD", "置信度阈值必须位于 0 到 1 之间");
    }
    try
    {
        if (request.pngImage) validatePngImage(*request.pngImage);
        else validateRawImage(*request.image);
    }
    catch (const std::invalid_argument& error)
    {
        throw InvalidInferenceRequest(request.pngImage ? "INVALID_PNG_IMAGE" : "INVALID_RAW_IMAGE", error.what());
    }
    std::shared_ptr<Impl::Generation> generation;
    {
        std::lock_guard<std::mutex> lock(impl_->mutex);
        generation = impl_->activeGeneration;
        if (impl_->stopping || !generation) throw BackendNotReady("推理模型尚未就绪");
        if (generation->modelIndex != static_cast<std::size_t>(found - impl_->models.begin()))
            throw InvalidInferenceRequest("MODEL_NOT_ACTIVE", "请求的模型版本不是当前选择，请先切换模型");
        if (!impl_->modelReady(*generation)) throw BackendNotReady("推理模型尚未就绪");
        if (impl_->inFlight >= impl_->runtime.maxInFlight) throw InferenceOverloaded("推理在途请求已满");
        ++impl_->inFlight;
        ++generation->inFlight;
    }
    struct Admission
    {
        Impl& impl;
        const std::shared_ptr<Impl::Generation>& generation;
        ~Admission() { impl.releaseAdmission(generation); }
    } admission{*impl_, generation};
    InferenceResult result;
    result.confidenceThreshold = request.confidenceThreshold;
    result.timings.validationMilliseconds = milliseconds(started);
    const auto preprocessStarted = Clock::now();
    PreparedImage prepared = request.pngImage ? prepareImage(*request.pngImage, found->inputSize)
                                              : prepareImage(*request.image, found->inputSize);
    result.timings.preprocessMilliseconds = milliseconds(preprocessStarted);
    const auto tritonStarted = Clock::now();
    std::vector<float> output = impl_->inferTensor(std::move(prepared.tensor), *generation, requestId);
    // 此字段在 HTTP 层以 tritonRequest 输出，包含 Triton 排队/复制/执行，不是纯 GPU 时间。
    result.timings.inferenceMilliseconds = milliseconds(tritonStarted);
    const auto postprocessStarted = Clock::now();
    result.detections = decodeDetections(output, prepared.letterbox, request.confidenceThreshold);
    for (Detection& detection : result.detections)
    {
        const auto name = found->classNames.find(detection.classId);
        if (name == found->classNames.end()) throw std::runtime_error("模型输出了未配置名称的类别 ID");
        detection.className = name->second;
    }
    result.timings.postprocessMilliseconds = milliseconds(postprocessStarted);
    result.elapsedMilliseconds = milliseconds(started);
    result.timings.totalMilliseconds = result.elapsedMilliseconds;
    return result;
}

Json BackendService::status() const
{
    std::lock_guard<std::mutex> lock(impl_->mutex);
    bool live = false;
    discardError(TRITONSERVER_ServerIsLive(impl_->server.get(), &live));
    Json models = Json::array();
    std::map<std::string, std::size_t> groups;
    for (std::size_t index = 0; index < impl_->models.size(); ++index)
    {
        const auto& model = impl_->models[index];
        const auto group = groups.emplace(model.id, models.size());
        if (group.second) models.push_back(Json{{"id", model.id}, {"versions", Json::array()}});
        models[group.first->second]["versions"].push_back(impl_->describe(index));
    }
    Json retiring = Json::array();
    for (const auto& generation : impl_->retiring)
    {
        const auto& model = impl_->models[generation->modelIndex];
        retiring.push_back(Json{{"id", model.id}, {"version", model.externalVersion},
            {"generation", generation->number}, {"inFlight", generation->inFlight},
            {"unloading", generation->unloading}, {"cleanupError", generation->cleanupError}});
    }
    const bool ready = !impl_->stopping && impl_->activeGeneration && impl_->modelReady(*impl_->activeGeneration);
    auto activeModel = impl_->selected();
    if (impl_->activeGeneration)
    {
        try
        {
            TRITONSERVER_Message* message = nullptr;
            const auto& generation = *impl_->activeGeneration;
            const auto& model = impl_->models[generation.modelIndex];
            check(TRITONSERVER_ServerModelStatistics(impl_->server.get(), generation.name.c_str(),
                  model.tritonVersion, &message), "查询模型执行统计");
            activeModel["executionStatistics"] = readMessage(message);
        }
        catch (const std::exception& error)
        {
            activeModel["executionStatistics"] = nullptr;
            activeModel["statisticsError"] = error.what();
        }
    }
    return Json{{"ok", true}, {"service", "triton-cpp-backend"}, {"serviceAlive", true},
                {"state", impl_->state}, {"modelReady", ready}, {"tritonLive", live},
                {"inFlight", impl_->inFlight}, {"lastError", impl_->lastError},
                {"switching", impl_->pendingModel.has_value()},
                {"pendingModel", impl_->pendingModel ? impl_->describe(*impl_->pendingModel) : Json(nullptr)},
                {"retiringModels", retiring}, {"activeModel", std::move(activeModel)},
                {"apiVersion", Json{{"major", impl_->apiMajor}, {"minor", impl_->apiMinor}}},
                {"triton", impl_->serverMetadata}, {"models", models},
                {"timingSemantics", "tritonRequest includes backend queueing, transfers and execution"}};
}

} // namespace triton_backend
