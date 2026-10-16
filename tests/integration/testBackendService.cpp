#include "services/impl/BackendService.h"

#include <gtest/gtest.h>

#include <opencv2/imgcodecs.hpp>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <future>
#include <limits>
#include <memory>
#include <string>
#include <thread>
#include <unistd.h>

#ifndef TEST_MODEL_REPOSITORY
#define TEST_MODEL_REPOSITORY "/opt/triton_back/models"
#endif

using namespace triton_backend;

namespace
{

RuntimeConfig runtimeConfig(std::size_t maxInFlight = 8)
{
    const char* repository = std::getenv("TRITON_TEST_REPOSITORY");
    const char* root = std::getenv("TRITON_TEST_ROOT");
    return RuntimeConfig{root == nullptr ? "/opt/triton/2.71.0" : root,
                         repository == nullptr ? TEST_MODEL_REPOSITORY : repository,
                         maxInFlight, 30};
}

#ifndef TEST_CONFIG_ROOT
#define TEST_CONFIG_ROOT "config"
#endif
#ifndef TEST_FIXTURE_ROOT
#define TEST_FIXTURE_ROOT "tests/fixtures"
#endif

std::vector<ModelConfig> configuredModels()
{
    std::ifstream input(std::filesystem::path(TEST_CONFIG_ROOT) / "models.json");
    if (!input) throw std::runtime_error("缺少测试模型配置 models.json");
    const auto document = nlohmann::json::parse(input);
    std::vector<ModelConfig> models;
    for (const auto& item : document.at("models"))
    {
        ModelConfig model{item.at("id"), item.at("businessVersion"), item.at("tritonName"),
                          item.at("tritonVersion"), item.at("inputSize"), {}};
        for (auto name = item.at("classNames").begin(); name != item.at("classNames").end(); ++name)
        {
            model.classNames.emplace(std::stoi(name.key()), name.value().get<std::string>());
        }
        model.instanceCount = item.value("instanceCount", std::size_t{1});
        if (item.contains("batching"))
        {
            const auto& batching = item.at("batching");
            model.batching.maxBatchSize = batching.value("maxBatchSize", std::size_t{0});
            model.batching.preferredBatchSizes = batching.value("preferredBatchSizes", std::vector<std::size_t>{});
            model.batching.maxQueueDelayMicroseconds = batching.value("maxQueueDelayMicroseconds", std::uint64_t{0});
        }
        models.push_back(std::move(model));
    }
    validateModelCatalog(models);
    return models;
}

const ModelConfig& configuredModel(
    const std::vector<ModelConfig>& models, const std::string& id, const std::string& version)
{
    const auto found = std::find_if(models.begin(), models.end(), [&](const auto& model)
    { return model.id == id && model.externalVersion == version; });
    if (found == models.end()) throw std::runtime_error("缺少测试模型版本：" + id + "/" + version);
    return *found;
}

std::vector<ModelConfig> legacyModels()
{
    const auto models = configuredModels();
    return {configuredModel(models, "best", "1.0.0"), configuredModel(models, "yolo26n", "1.0.0")};
}

RawImage realImage(const std::string& modelId)
{
    const std::filesystem::path fixtureRoot(TEST_FIXTURE_ROOT);
    std::ifstream metadataFile(fixtureRoot / "fixtures.json");
    if (!metadataFile) throw std::runtime_error("缺少测试像素描述 fixtures.json");
    const auto document = nlohmann::json::parse(metadataFile);
    for (const auto& fixture : document.at("fixtures"))
    {
        if (fixture.at("modelId") != modelId) continue;
        if (fixture.at("pixelFormat") != "BGR8Packed" || fixture.at("rowOrder") != "TopDown")
            throw std::runtime_error("GPU 测试样本应为 BGR8Packed TopDown");
        const auto relativePath = std::filesystem::path(fixture.at("rawFile").get<std::string>());
        if (relativePath.is_absolute() || relativePath.has_parent_path())
            throw std::runtime_error("GPU 测试样本应位于 fixtures 目录中");
        std::ifstream raw(fixtureRoot / relativePath, std::ios::binary);
        if (!raw) throw std::runtime_error("无法读取 GPU 测试像素文件");
        std::string pixels{std::istreambuf_iterator<char>(raw), std::istreambuf_iterator<char>()};
        RawImage image{fixture.at("width"), fixture.at("height"), PixelFormat::BGR8Packed,
                       fixture.at("strideBytes"), false,
                       std::make_shared<const std::string>(std::move(pixels))};
        validateRawImage(image);
        return image;
    }
    throw std::runtime_error("缺少模型的 GPU 测试像素");
}

void expectBaseline(const InferenceResult& result, const std::string& modelId)
{
    std::ifstream referenceFile(std::filesystem::path(TEST_FIXTURE_ROOT) / "expected-baseline.json");
    if (!referenceFile) throw std::runtime_error("缺少旧实现的检测参考结果");
    const auto document = nlohmann::json::parse(referenceFile);
    const auto& reference = document.at("models").at(modelId);
    ASSERT_EQ(result.detections.size(), reference.at("detectionCount").get<std::size_t>());
    ASSERT_GT(result.detections.size(), 0U);
    const auto& expectedDetections = reference.at("detections");
    ASSERT_EQ(result.detections.size(), expectedDetections.size());
    for (std::size_t index = 0; index < result.detections.size(); ++index)
    {
        const auto& actual = result.detections[index];
        const auto& expected = expectedDetections[index];
        EXPECT_EQ(actual.classId, expected.at("classId"));
        EXPECT_NEAR(actual.confidence, expected.at("confidence").get<float>(),
                    document.at("confidenceTolerance").get<float>());
        const auto& box = expected.at("box");
        const float tolerance = document.at("coordinateTolerance").get<float>();
        EXPECT_NEAR(actual.x1, box.at("x1").get<float>(), tolerance);
        EXPECT_NEAR(actual.y1, box.at("y1").get<float>(), tolerance);
        EXPECT_NEAR(actual.x2, box.at("x2").get<float>(), tolerance);
        EXPECT_NEAR(actual.y2, box.at("y2").get<float>(), tolerance);
    }
}

InferenceRequest requestFor(const RawImage& image)
{
    return InferenceRequest{{}, 0.25F, image};
}

InferenceRequest pngRequestFor(const RawImage& image)
{
    const cv::Mat pixels(image.height, image.width, CV_8UC3,
                         const_cast<char*>(image.data->data()), image.strideBytes);
    std::vector<unsigned char> encoded;
    if (!cv::imencode(".png", pixels, encoded)) throw std::runtime_error("无法编码测试 PNG");
    InferenceRequest request;
    request.pngImage = PngImage{image.width, image.height,
        std::make_shared<const std::string>(encoded.begin(), encoded.end())};
    return request;
}

const nlohmann::json& modelVersion(
    const nlohmann::json& status, const std::string& id, const std::string& version)
{
    for (const auto& model : status.at("models"))
        if (model.at("id") == id)
            for (const auto& item : model.at("versions"))
                if (item.at("version") == version) return item;
    throw std::runtime_error("状态缺少模型版本：" + id + "/" + version);
}

bool waitForRetirement(BackendService& service)
{
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(10);
    while (std::chrono::steady_clock::now() < deadline)
    {
        if (service.status().at("retiringModels").empty()) return true;
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    return false;
}

void expectValid(const InferenceResult& result, const RawImage& image, const ModelConfig& model)
{
    EXPECT_LE(result.detections.size(), 300U);
    EXPECT_GT(result.timings.inferenceMilliseconds, 0);
    EXPECT_GE(result.timings.totalMilliseconds, result.timings.inferenceMilliseconds);
    for (const auto& detection : result.detections)
    {
        EXPECT_EQ(detection.className, model.classNames.at(detection.classId));
        EXPECT_GE(detection.confidence, 0.25F);
        EXPECT_LE(detection.confidence, 1.0F);
        EXPECT_GE(detection.x1, 0);
        EXPECT_GE(detection.y1, 0);
        EXPECT_GT(detection.x2, detection.x1);
        EXPECT_GT(detection.y2, detection.y1);
        EXPECT_LE(detection.x2, image.width - 1);
        EXPECT_LE(detection.y2, image.height - 1);
    }
}

void expectEquivalent(const InferenceResult& result, const InferenceResult& reference)
{
    ASSERT_EQ(result.detections.size(), reference.detections.size());
    for (std::size_t index = 0; index < reference.detections.size(); ++index)
    {
        const auto& actual = result.detections[index];
        const auto& expected = reference.detections[index];
        EXPECT_EQ(actual.classId, expected.classId);
        EXPECT_EQ(actual.className, expected.className);
        EXPECT_NEAR(actual.confidence, expected.confidence, 1e-5F);
        EXPECT_NEAR(actual.x1, expected.x1, 1e-3F);
        EXPECT_NEAR(actual.y1, expected.y1, 1e-3F);
        EXPECT_NEAR(actual.x2, expected.x2, 1e-3F);
        EXPECT_NEAR(actual.y2, expected.y2, 1e-3F);
    }
}

nlohmann::json settledStatistics(BackendService& service, std::uint64_t minimumImages = 0)
{
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
    while (std::chrono::steady_clock::now() < deadline)
    {
        const auto status = service.status();
        const auto& active = status.at("activeModel");
        if (active.contains("statisticsError"))
            throw std::runtime_error(active.at("statisticsError").get<std::string>());
        const auto& models = active.at("executionStatistics").at("model_stats");
        if (models.size() != 1) throw std::runtime_error("活动模型应只有一份执行统计");
        const auto& statistics = models[0];
        std::uint64_t executions = 0;
        std::uint64_t images = 0;
        for (const auto& batch : statistics.at("batch_stats"))
        {
            const auto count = batch.at("compute_infer").at("count").get<std::uint64_t>();
            executions += count;
            images += batch.at("batch_size").get<std::uint64_t>() * count;
        }
        // Triton 在释放请求之后才报告整批统计，读取时等待最后一批补齐。
        if (status.at("inFlight") == 0 && images >= minimumImages && statistics.at("execution_count") == executions &&
            statistics.at("inference_count") == images)
            return statistics;
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    throw std::runtime_error("等待模型执行统计完成超时");
}

std::uint64_t executionsForBatch(const nlohmann::json& statistics, std::size_t size)
{
    for (const auto& batch : statistics.at("batch_stats"))
        if (batch.at("batch_size") == size)
            return batch.at("compute_infer").at("count").get<std::uint64_t>();
    return 0;
}

void expectTf32Reference(const InferenceResult& result, const nlohmann::json& references, std::size_t batchSize)
{
    const auto found = std::find_if(references.begin(), references.end(), [&](const auto& reference)
    {
        if (reference.at("batchSize") != batchSize || reference.at("detectionCount") != result.detections.size() ||
            reference.at("detections").size() != result.detections.size()) return false;
        for (std::size_t index = 0; index < result.detections.size(); ++index)
        {
            const auto& actual = result.detections[index];
            const auto& expected = reference.at("detections")[index];
            if (actual.classId != expected.at("classId") || actual.className != expected.at("className") ||
                !std::isfinite(actual.confidence) ||
                std::abs(actual.confidence - expected.at("confidence").template get<double>()) > 1e-5)
                return false;
            const auto& box = expected.at("box");
            if (!std::isfinite(actual.x1) || !std::isfinite(actual.y1) ||
                !std::isfinite(actual.x2) || !std::isfinite(actual.y2) ||
                std::abs(actual.x1 - box.at("x1").template get<double>()) > 1e-3 ||
                std::abs(actual.y1 - box.at("y1").template get<double>()) > 1e-3 ||
                std::abs(actual.x2 - box.at("x2").template get<double>()) > 1e-3 ||
                std::abs(actual.y2 - box.at("y2").template get<double>()) > 1e-3)
                return false;
        }
        return true;
    });
    // 请求提交顺序不等于合批位置，只匹配此实际批量下的直接 TensorRT 参考。
    EXPECT_NE(found, references.end()) << "未匹配 TF32 直接参考，实际批量为 " << batchSize;
}

TEST(BackendRuntimeTest, SelectsEachModelInfersRealPicturesAndRestarts)
{
    const auto models = legacyModels();
    ASSERT_EQ(models.size(), 2U);
    BackendService service(runtimeConfig(), models);
    EXPECT_FALSE(service.isReady());
    EXPECT_EQ(service.status().at("state"), "stopped");
    EXPECT_TRUE(service.status().at("serviceAlive"));
    EXPECT_TRUE(service.status().at("tritonLive"));
    EXPECT_TRUE(service.status().at("activeModel").is_null());
    EXPECT_TRUE(service.status().contains("apiVersion"));
    EXPECT_TRUE(service.status().contains("triton"));
    EXPECT_EQ(modelVersion(service.status(), models[0].id, models[0].externalVersion)
                  .at("metadataSource"), "configuration");
    const auto unselected = service.status();
    const auto& configured = modelVersion(unselected, models[0].id, models[0].externalVersion);
    EXPECT_EQ(configured.at("inputs")[0].at("name"), "images");
    EXPECT_EQ(configured.at("inputs")[0].at("datatype"), "FP32");
    EXPECT_EQ(configured.at("inputs")[0].at("shape"),
              nlohmann::json::array({1, 3, models[0].inputSize, models[0].inputSize}));
    EXPECT_EQ(configured.at("outputs")[0].at("name"), "output0");
    EXPECT_EQ(configured.at("outputs")[0].at("shape"), nlohmann::json::array({1, 300, 6}));
    service.selectModel({models[0].id, models[0].externalVersion});
    service.selectModel({models[0].id, models[0].externalVersion});
    ASSERT_TRUE(service.isReady());
    EXPECT_EQ(service.status().at("state"), "running");
    EXPECT_EQ(modelVersion(service.status(), models[0].id, models[0].externalVersion)
                  .at("metadataSource"), "loaded_model");
    std::vector<RawImage> images;
    std::vector<InferenceResult> serialResults;
    for (const auto& model : models)
    {
        service.selectModel(ModelSelection{model.id, model.externalVersion});
        const RawImage image = realImage(model.id);
        auto result = service.infer(requestFor(image), model.id, model.externalVersion, "real-" + model.id);
        expectValid(result, image, model);
        expectBaseline(result, model.id);
        const auto pngResult = service.infer(pngRequestFor(image), model.id, model.externalVersion,
                                             "png-" + model.id);
        expectValid(pngResult, image, model);
        expectBaseline(pngResult, model.id);
        images.push_back(image);
        serialResults.push_back(std::move(result));
    }
    std::vector<std::future<InferenceResult>> concurrent;
    for (std::size_t index = 0; index < 4; ++index)
    {
        concurrent.push_back(std::async(std::launch::async, [&, index]
        {
            const std::size_t modelIndex = models.size() - 1;
            const auto& model = models[modelIndex];
            return service.infer(requestFor(images[modelIndex]), model.id, model.externalVersion,
                                 "selected-model-" + std::to_string(index));
        }));
    }
    for (std::size_t index = 0; index < concurrent.size(); ++index)
    {
        const auto result = concurrent[index].get();
        const auto& serial = serialResults.back();
        ASSERT_EQ(result.detections.size(), serial.detections.size());
        for (std::size_t detection = 0; detection < serial.detections.size(); ++detection)
        {
            const auto& actual = result.detections[detection];
            const auto& expected = serial.detections[detection];
            EXPECT_EQ(actual.classId, expected.classId);
            EXPECT_EQ(actual.className, expected.className);
            EXPECT_NEAR(actual.confidence, expected.confidence, 1e-6F);
            EXPECT_NEAR(actual.x1, expected.x1, 1e-3F);
            EXPECT_NEAR(actual.y1, expected.y1, 1e-3F);
            EXPECT_NEAR(actual.x2, expected.x2, 1e-3F);
            EXPECT_NEAR(actual.y2, expected.y2, 1e-3F);
        }
    }
    const auto image = realImage("best");
    EXPECT_THROW(service.infer(requestFor(image), "unknown", "1.0.0", "bad-model"), InvalidInferenceRequest);
    EXPECT_THROW(service.infer(requestFor(image), models[0].id, "unknown", "bad-version"), InvalidInferenceRequest);
    auto invalid = requestFor(image);
    invalid.confidenceThreshold = std::numeric_limits<float>::quiet_NaN();
    EXPECT_THROW(service.infer(invalid, models[0].id, models[0].externalVersion, "bad-threshold"), InvalidInferenceRequest);
    service.stop();
    service.stop();
    EXPECT_FALSE(service.isReady());
    EXPECT_EQ(service.status().at("state"), "stopped");
    EXPECT_EQ(service.status().at("inFlight"), 0);
    EXPECT_THROW(service.infer(requestFor(image), models[0].id, models[0].externalVersion, "stopped"), BackendNotReady);
    service.selectModel({models[0].id, models[0].externalVersion});
    EXPECT_TRUE(service.isReady());
}

TEST(BackendRuntimeTest, MissingModelFileKeepsCoreAvailableAndCanBeStopped)
{
    auto model = configuredModels().front();
    model.tritonName = "missing_model_for_lifecycle_test";
    BackendService service(runtimeConfig(), {model});
    EXPECT_THROW(service.selectModel({model.id, model.externalVersion}), InvalidInferenceRequest);
    EXPECT_EQ(service.status().at("state"), "stopped");
    EXPECT_FALSE(service.isReady());
    EXPECT_TRUE(service.status().at("tritonLive"));
    EXPECT_EQ(modelVersion(service.status(), model.id, model.externalVersion).at("metadataSource"), "configuration");
    EXPECT_FALSE(modelVersion(service.status(), model.id, model.externalVersion).at("available"));
    EXPECT_THROW(service.selectModel({model.id, model.externalVersion}), InvalidInferenceRequest);
    service.stop();
    EXPECT_EQ(service.status().at("state"), "stopped");
}

TEST(BackendRuntimeTest, ExplicitVersionsSwitchAndFailedLoadingLeavesPreviousGenerationUntouched)
{
    struct Repository
    {
        std::filesystem::path path = std::filesystem::temp_directory_path() /
            ("triton-version-test-" + std::to_string(getpid()) + "-" +
             std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
        ~Repository() { std::filesystem::remove_all(path); }
    } repository;
    auto models = legacyModels();
    const auto original = runtimeConfig();
    for (const auto& model : models)
    {
        const auto source = std::filesystem::path(original.repository) / model.tritonName;
        const auto target = repository.path / model.tritonName;
        std::filesystem::create_directories(target / "1");
        std::filesystem::copy_file(source / "config.pbtxt", target / "config.pbtxt");
        std::filesystem::copy_file(source / "1/model.plan", target / "1/model.plan");
    }
    const auto root = repository.path / "best";
    std::filesystem::create_directories(root / "2");
    std::filesystem::copy_file(root / "1/model.plan", root / "2/model.plan");
    std::filesystem::create_directories(root / "3");
    std::ofstream(root / "3/model.plan") << "invalid-engine";
    for (int version : {2, 3, 4})
    {
        auto model = models.front();
        model.externalVersion = std::to_string(version) + ".0.0";
        model.tritonVersion = version;
        models.push_back(model);
    }
    const auto mismatchSource = repository.path / "yolo26n";
    const auto mismatchTarget = repository.path / "incompatible_shape";
    std::filesystem::create_directories(mismatchTarget / "1");
    std::filesystem::copy_file(mismatchSource / "1/model.plan", mismatchTarget / "1/model.plan");
    std::ifstream mismatchInput(mismatchSource / "config.pbtxt");
    std::string mismatchConfig{std::istreambuf_iterator<char>(mismatchInput), std::istreambuf_iterator<char>()};
    const std::string originalName = "name: \"yolo26n\"";
    const auto nameOffset = mismatchConfig.find(originalName);
    ASSERT_NE(nameOffset, std::string::npos);
    mismatchConfig.replace(nameOffset, originalName.size(), "name: \"incompatible_shape\"");
    std::ofstream(mismatchTarget / "config.pbtxt") << mismatchConfig;
    auto incompatible = models[1];
    incompatible.id = "incompatible_shape";
    incompatible.tritonName = "incompatible_shape";
    incompatible.inputSize = 960;
    models.push_back(incompatible);
    auto settings = original;
    settings.repository = repository.path.string();
    BackendService service(settings, models);
    const auto catalog = service.status();
    EXPECT_TRUE(catalog.at("activeModel").is_null());
    ASSERT_EQ(catalog.at("models")[0].at("versions").size(), 4U);
    EXPECT_FALSE(catalog.at("models")[0].at("versions")[3].at("available"));
    const auto image = realImage("best");
    service.selectModel({"best", "1.0.0"});
    expectBaseline(service.infer(requestFor(image), "best", "1.0.0", "version-1"), "best");
    service.selectModel({"best", "2.0.0"});
    const auto generation = service.status().at("activeModel").at("generation");
    service.selectModel({"best", "2.0.0"});
    EXPECT_EQ(service.status().at("activeModel").at("generation"), generation);
    ASSERT_TRUE(service.isReady());
    EXPECT_EQ(service.status().at("activeModel").at("tritonVersion"), 2);
    expectBaseline(service.infer(requestFor(image), "best", "2.0.0", "version-2"), "best");
    EXPECT_THROW(service.infer(requestFor(image), "best", "1.0.0", "old-frame"), InvalidInferenceRequest);
    for (const auto& selection : {ModelSelection{"unknown", "1.0.0"}, ModelSelection{"best", "unknown"}, ModelSelection{"best", "4.0.0"}})
        EXPECT_THROW(service.selectModel(selection), InvalidInferenceRequest);
    EXPECT_EQ(service.status().at("activeModel").at("version"), "2.0.0");
    EXPECT_THROW(service.selectModel({"best", "3.0.0"}), std::runtime_error);
    EXPECT_TRUE(service.isReady());
    EXPECT_EQ(service.status().at("activeModel").at("version"), "2.0.0");
    EXPECT_EQ(service.status().at("activeModel").at("generation"), generation);
    EXPECT_FALSE(service.status().at("switching"));
    EXPECT_TRUE(service.status().at("pendingModel").is_null());
    EXPECT_FALSE(service.status().at("lastError").get<std::string>().empty());
    expectBaseline(service.infer(requestFor(image), "best", "2.0.0", "after-failed-switch"), "best");
    EXPECT_THROW(service.selectModel({"incompatible_shape", "1.0.0"}), std::runtime_error);
    EXPECT_EQ(service.status().at("activeModel").at("generation"), generation);
    EXPECT_TRUE(service.isReady());
    EXPECT_FALSE(service.status().at("switching"));
    EXPECT_TRUE(service.status().at("pendingModel").is_null());
    EXPECT_FALSE(service.status().at("lastError").get<std::string>().empty());
    expectBaseline(service.infer(requestFor(image), "best", "2.0.0", "after-shape-mismatch"), "best");
    service.selectModel({"yolo26n", "1.0.0"});
    EXPECT_EQ(service.status().at("activeModel").at("id"), "yolo26n");
    int ready = 0;
    const auto status = service.status();
    for (const auto& model : status.at("models"))
        for (const auto& version : model.at("versions"))
            ready += version.at("active").get<bool>() && version.at("ready").get<bool>();
    EXPECT_EQ(ready, 1);
    expectBaseline(service.infer(requestFor(realImage("yolo26n")), "yolo26n", "1.0.0", "different-model"), "yolo26n");
    service.stop();
    service.selectModel({"yolo26n", "1.0.0"});
    EXPECT_EQ(service.status().at("activeModel").at("id"), "yolo26n");
}

TEST(BackendRuntimeTest, SwitchingKeepsOldModelServingThenRetiresAcceptedRequests)
{
    const auto models = configuredModels();
    BackendService service(runtimeConfig(32), models);
    service.selectModel({"best", "1.0.0"});
    const auto oldGeneration = service.status().at("activeModel").at("generation");
    const auto image = realImage("best");
    expectBaseline(service.infer(requestFor(image), "best", "1.0.0", "before-switch"), "best");
    auto switching = std::async(std::launch::async, [&]
    { service.selectModel({"yolo26n", "1.0.0"}); });
    bool observedLoading = false;
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(10);
    while (std::chrono::steady_clock::now() < deadline)
    {
        if (service.status().at("switching")) { observedLoading = true; break; }
        if (switching.wait_for(std::chrono::milliseconds(0)) == std::future_status::ready) break;
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    ASSERT_TRUE(observedLoading);
    const auto loadingStatus = service.status();
    EXPECT_EQ(loadingStatus.at("activeModel").at("generation"), oldGeneration);
    EXPECT_EQ(loadingStatus.at("pendingModel").at("id"), "yolo26n");
    EXPECT_TRUE(loadingStatus.at("modelReady"));
    EXPECT_TRUE(service.isReady());
    expectBaseline(service.infer(requestFor(image), "best", "1.0.0", "while-loading"), "best");
    EXPECT_TRUE(service.status().at("switching"));

    std::atomic<bool> done{false};
    struct StopRequests
    {
        std::atomic<bool>& value;
        ~StopRequests() { value.store(true); }
    };
    std::vector<std::future<int>> oldRequests;
    StopRequests stopRequests{done};
    for (int index = 0; index < 16; ++index)
    {
        oldRequests.push_back(std::async(std::launch::async, [&, index]
        {
            int completed = 0;
            while (!done.load())
            {
                try
                {
                    const auto result = service.infer(requestFor(image), "best", "1.0.0",
                        "switch-old-" + std::to_string(index) + "-" + std::to_string(completed));
                    expectBaseline(result, "best");
                    ++completed;
                }
                catch (const InvalidInferenceRequest& error)
                {
                    EXPECT_EQ(error.code(), "MODEL_NOT_ACTIVE");
                    break;
                }
            }
            return completed;
        }));
    }
    bool observedRetiring = false;
    std::size_t retiringInFlight = 0;
    const auto switchDeadline = std::chrono::steady_clock::now() + std::chrono::seconds(30);
    while (std::chrono::steady_clock::now() < switchDeadline)
    {
        const auto status = service.status();
        for (const auto& retiring : status.at("retiringModels"))
        {
            if (retiring.at("generation") != oldGeneration) continue;
            observedRetiring = true;
            EXPECT_EQ(retiring.at("id"), "best");
            EXPECT_EQ(retiring.at("version"), "1.0.0");
            EXPECT_TRUE(retiring.at("cleanupError").get<std::string>().empty());
            retiringInFlight = std::max(retiringInFlight, retiring.at("inFlight").get<std::size_t>());
        }
        if (status.at("activeModel").at("id") == "yolo26n") break;
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    done.store(true);
    EXPECT_EQ(switching.wait_for(std::chrono::seconds(10)), std::future_status::ready);
    switching.get();
    EXPECT_TRUE(observedRetiring);
    EXPECT_GT(retiringInFlight, 0U);
    RecordProperty("retiringInFlightAtCutover", std::to_string(retiringInFlight));
    EXPECT_EQ(service.status().at("activeModel").at("id"), "yolo26n");
    EXPECT_NE(service.status().at("activeModel").at("generation"), oldGeneration);
    EXPECT_TRUE(service.isReady());
    expectBaseline(service.infer(requestFor(realImage("yolo26n")), "yolo26n", "1.0.0",
                                "after-cutover"), "yolo26n");
    int completed = 0;
    for (auto& request : oldRequests) completed += request.get();
    EXPECT_GT(completed, 0);
    RecordProperty("completedOldRequests", completed);
    EXPECT_TRUE(waitForRetirement(service));
    EXPECT_EQ(service.status().at("inFlight"), 0);
    EXPECT_THROW(service.infer(requestFor(image), "best", "1.0.0", "after-cutover-old"), InvalidInferenceRequest);
}

TEST(BackendRuntimeTest, ConcurrentRequestsAreBoundedAndStopDrainsAcceptedRequest)
{
    const auto models = configuredModels();
    BackendService service(runtimeConfig(1), models);
    service.selectModel({models[0].id, models[0].externalVersion});
    const auto image = realImage("best");
    std::promise<void> release;
    const auto barrier = release.get_future().share();
    std::vector<std::future<bool>> requests;
    for (int index = 0; index < 8; ++index)
    {
        requests.push_back(std::async(std::launch::async, [&, index]
        {
            barrier.wait();
            try
            {
                expectValid(service.infer(requestFor(image), models[0].id, models[0].externalVersion,
                                           "concurrent-" + std::to_string(index)), image, models[0]);
                return true;
            }
            catch (const InferenceOverloaded&) { return false; }
        }));
    }
    release.set_value();
    int accepted = 0;
    for (auto& request : requests) accepted += request.get() ? 1 : 0;
    EXPECT_GT(accepted, 0);
    EXPECT_LT(accepted, 8);
    EXPECT_EQ(service.status().at("inFlight"), 0);
    auto active = std::async(std::launch::async, [&]
    {
        return service.infer(requestFor(image), models[0].id, models[0].externalVersion, "drain");
    });
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(2);
    bool observedActive = false;
    while (std::chrono::steady_clock::now() < deadline)
    {
        if (service.status().at("inFlight") == 1) { observedActive = true; break; }
        if (active.wait_for(std::chrono::milliseconds(0)) == std::future_status::ready) break;
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    ASSERT_TRUE(observedActive);
    service.stop();
    EXPECT_EQ(active.wait_for(std::chrono::milliseconds(0)), std::future_status::ready);
    expectValid(active.get(), image, models[0]);
    EXPECT_EQ(service.status().at("inFlight"), 0);
    EXPECT_EQ(service.status().at("state"), "stopped");
    EXPECT_THROW(service.infer(requestFor(image), models[0].id, models[0].externalVersion, "after-drain"), BackendNotReady);
}

TEST(BackendRuntimeTest, DynamicBatchingPreservesSingleImageResultsAndFailedSwitchKeepsActiveModel)
{
    struct Repository
    {
        std::filesystem::path path = std::filesystem::temp_directory_path() /
            ("triton-batching-test-" + std::to_string(getpid()) + "-" +
             std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
        ~Repository() { std::filesystem::remove_all(path); }
    } repository;
    auto models = configuredModels();
    const auto configured = configuredModel(models, "best", "2.0.0");
    ASSERT_EQ(configured.tritonVersion, 2);
    ASSERT_EQ(configured.batching.maxBatchSize, 8U);
    ASSERT_EQ(configured.batching.preferredBatchSizes, std::vector<std::size_t>{4});
    ASSERT_EQ(configured.instanceCount, 1U);
    // 为测试线程和 PNG 解码留出余量；通过真实批量统计验证合批，不断言精确等待时间。
    for (auto& model : models)
        if (model.id == "best" && model.externalVersion == "2.0.0")
            model.batching.maxQueueDelayMicroseconds = 100000;
    const auto batching = configuredModel(models, "best", "2.0.0");
    const auto original = runtimeConfig();
    std::filesystem::create_directory(repository.path);
    for (const auto& model : models)
    {
        const auto target = repository.path / model.tritonName;
        if (!std::filesystem::exists(target))
            std::filesystem::create_directory_symlink(
                std::filesystem::path(original.repository) / model.tritonName, target);
    }
    auto invalid = batching;
    invalid.id = "invalid_batch_candidate";
    invalid.tritonName = "invalid_batch_candidate";
    invalid.externalVersion = "invalid";
    invalid.tritonVersion = 1;
    std::filesystem::create_directories(repository.path / invalid.tritonName / "1");
    std::ofstream(repository.path / invalid.tritonName / "1/model.plan") << "invalid-engine";
    models.push_back(invalid);
    auto settings = original;
    settings.repository = repository.path.string();
    BackendService service(settings, models);
    const auto image = realImage("best");
    service.selectModel({"best", "1.0.0"});
    const auto previousGeneration = service.status().at("activeModel").at("generation");
    expectBaseline(service.infer(requestFor(image), "best", "1.0.0", "before-batching"), "best");
    service.selectModel({batching.id, batching.externalVersion});
    const auto selected = service.status();
    const auto generation = selected.at("activeModel").at("generation");
    EXPECT_NE(generation, previousGeneration);
    EXPECT_TRUE(selected.at("modelReady"));
    EXPECT_EQ(selected.at("activeModel").at("inputs")[0].at("shape"),
              nlohmann::json::array({-1, 3, batching.inputSize, batching.inputSize}));
    EXPECT_EQ(selected.at("activeModel").at("outputs")[0].at("shape"), nlohmann::json::array({-1, 300, 6}));
    EXPECT_EQ(selected.at("activeModel").at("batching").at("preferredBatchSizes"), nlohmann::json::array({4}));
    ASSERT_TRUE(waitForRetirement(service));

    const cv::Mat source(image.height, image.width, CV_8UC3,
                         const_cast<char*>(image.data->data()), image.strideBytes);
    std::vector<RawImage> images;
    std::vector<InferenceRequest> requests;
    std::vector<InferenceResult> reference;
    const auto beforeSerial = settledStatistics(service, 1).at("inference_count").get<std::uint64_t>();
    for (int index = 0; index < 4; ++index)
    {
        cv::Mat pixels;
        if (index == 0) pixels = source.clone();
        else cv::flip(source, pixels, index == 1 ? 1 : index == 2 ? 0 : -1);
        RawImage variant{image.width, image.height, PixelFormat::BGR8Packed,
                         pixels.step[0], false,
                         std::make_shared<const std::string>(reinterpret_cast<const char*>(pixels.data),
                                                             pixels.total() * pixels.elemSize())};
        images.push_back(variant);
        requests.push_back(pngRequestFor(variant));
        reference.push_back(service.infer(requests.back(), batching.id, batching.externalVersion,
                                          "batch-serial-" + std::to_string(index)));
        expectValid(reference.back(), variant, batching);
    }
    const auto serialStatistics = settledStatistics(service, beforeSerial + images.size());
    EXPECT_EQ(serialStatistics.at("inference_count"), serialStatistics.at("execution_count"));
    EXPECT_EQ(executionsForBatch(serialStatistics, 1), serialStatistics.at("execution_count").get<std::uint64_t>());

    const auto runBurst = [&](std::size_t count, int round)
    {
        std::promise<void> release;
        const auto barrier = release.get_future().share();
        std::vector<std::promise<void>> arrived(count);
        std::vector<std::future<InferenceResult>> pending;
        for (std::size_t index = 0; index < count; ++index)
        {
            pending.push_back(std::async(std::launch::async, [&, index]
            {
                arrived[index].set_value();
                barrier.wait();
                return service.infer(requests[index], batching.id, batching.externalVersion,
                    "batch-burst-" + std::to_string(round) + "-" + std::to_string(index));
            }));
        }
        for (auto& arrival : arrived) arrival.get_future().wait();
        release.set_value();
        for (std::size_t index = 0; index < count; ++index)
        {
            const auto result = pending[index].get();
            expectValid(result, images[index], batching);
            expectEquivalent(result, reference[index]);
        }
    };
    auto before = serialStatistics;
    for (const auto count : {2U, 3U})
    {
        runBurst(count, static_cast<int>(count));
        const auto after = settledStatistics(service, before.at("inference_count").get<std::uint64_t>() + count);
        EXPECT_EQ(after.at("inference_count").get<std::uint64_t>() - before.at("inference_count").get<std::uint64_t>(), count);
        EXPECT_GT(executionsForBatch(after, count), executionsForBatch(before, count));
        before = after;
    }
    const auto beforeFour = before;
    for (int round = 0; round < 3; ++round) runBurst(4, 10 + round);
    const auto afterFour = settledStatistics(service, beforeFour.at("inference_count").get<std::uint64_t>() + 12);
    EXPECT_EQ(afterFour.at("inference_count").get<std::uint64_t>() -
              beforeFour.at("inference_count").get<std::uint64_t>(), 12U);
    const auto batchFourExecutions = executionsForBatch(afterFour, 4) - executionsForBatch(beforeFour, 4);
    EXPECT_GT(batchFourExecutions, 0U);
    EXPECT_GT(afterFour.at("execution_count").get<std::uint64_t>(), beforeFour.at("execution_count").get<std::uint64_t>());
    RecordProperty("dynamicBatch4Executions", std::to_string(batchFourExecutions));
    RecordProperty("batchQueueDelayMicroseconds", std::to_string(batching.batching.maxQueueDelayMicroseconds));

    EXPECT_THROW(service.selectModel({invalid.id, invalid.externalVersion}), std::runtime_error);
    const auto failed = service.status();
    EXPECT_EQ(failed.at("activeModel").at("generation"), generation);
    EXPECT_EQ(failed.at("activeModel").at("version"), batching.externalVersion);
    EXPECT_TRUE(failed.at("modelReady"));
    EXPECT_FALSE(failed.at("switching"));
    EXPECT_TRUE(failed.at("pendingModel").is_null());
    EXPECT_FALSE(failed.at("lastError").get<std::string>().empty());
    expectEquivalent(service.infer(requests[0], batching.id, batching.externalVersion, "batch-after-failure"), reference[0]);
    EXPECT_TRUE(waitForRetirement(service));
    service.selectModel({"best", "1.0.0"});
    EXPECT_EQ(service.status().at("activeModel").at("batching").at("maxBatchSize"), 0);
    expectBaseline(service.infer(requestFor(image), "best", "1.0.0", "back-to-legacy"), "best");
    EXPECT_TRUE(waitForRetirement(service));
}

TEST(BackendRuntimeTest, Tf32BatchSizesMatchDirectReferencesAndFailedSwitchKeepsActiveModel)
{
    std::ifstream referenceFile(std::filesystem::path(TEST_FIXTURE_ROOT) / "expected-best-tf32.json");
    ASSERT_TRUE(referenceFile) << "缺少 TF32 Engine 的直接 TensorRT 参考";
    const auto reference = nlohmann::json::parse(referenceFile);
    ASSERT_EQ(reference.at("modelId"), "best");
    ASSERT_EQ(reference.at("businessVersion"), "3.0.0");
    ASSERT_EQ(reference.at("tritonVersion"), 3);
    ASSERT_EQ(reference.at("inputSize"), 960);
    ASSERT_DOUBLE_EQ(reference.at("confidenceThreshold").get<double>(), 0.25);
    ASSERT_DOUBLE_EQ(reference.at("confidenceTolerance").get<double>(), 1e-5);
    ASSERT_DOUBLE_EQ(reference.at("coordinateTolerance").get<double>(), 1e-3);
    const auto& references = reference.at("references");
    ASSERT_EQ(references.size(), 36U);
    for (std::size_t batchSize = 1; batchSize <= 8; ++batchSize)
        for (std::size_t position = 0; position < batchSize; ++position)
            ASSERT_EQ(std::count_if(references.begin(), references.end(), [&](const auto& item)
                { return item.at("batchSize") == batchSize && item.at("batchPosition") == position; }), 1);

    const auto original = runtimeConfig();
    std::ifstream manifestFile(std::filesystem::path(original.repository) / "manifest.json");
    ASSERT_TRUE(manifestFile);
    const auto manifest = nlohmann::json::parse(manifestFile);
    const auto manifestModel = std::find_if(manifest.at("models").begin(), manifest.at("models").end(),
        [](const auto& model) { return model.at("id") == "best" && model.at("tritonVersion") == 3; });
    ASSERT_NE(manifestModel, manifest.at("models").end());
    ASSERT_EQ(reference.at("engineSha256"), manifestModel->at("sha256"));
    std::ifstream fixtureFile(std::filesystem::path(TEST_FIXTURE_ROOT) / "fixtures.json");
    ASSERT_TRUE(fixtureFile);
    const auto fixtures = nlohmann::json::parse(fixtureFile);
    const auto fixture = std::find_if(fixtures.at("fixtures").begin(), fixtures.at("fixtures").end(),
        [](const auto& image) { return image.at("modelId") == "best"; });
    ASSERT_NE(fixture, fixtures.at("fixtures").end());
    ASSERT_EQ(reference.at("rawSha256"), fixture->at("sha256"));
    const auto image = realImage("best");
    ASSERT_EQ(reference.at("width"), image.width);
    ASSERT_EQ(reference.at("height"), image.height);
    const auto request = pngRequestFor(image);

    struct Repository
    {
        std::filesystem::path path = std::filesystem::temp_directory_path() /
            ("triton-tf32-test-" + std::to_string(getpid()) + "-" +
             std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
        ~Repository() { std::filesystem::remove_all(path); }
    } repository;
    const auto configured = configuredModels();
    const auto tf32 = configuredModel(configured, "best", "3.0.0");
    ASSERT_EQ(tf32.tritonVersion, 3);
    ASSERT_EQ(tf32.batching.maxBatchSize, 8U);
    ASSERT_EQ(tf32.batching.preferredBatchSizes, std::vector<std::size_t>{4});
    ASSERT_EQ(tf32.instanceCount, 1U);
    std::filesystem::create_directory(repository.path);
    for (const auto& model : configured)
    {
        const auto target = repository.path / model.tritonName;
        if (!std::filesystem::exists(target))
            std::filesystem::create_directory_symlink(
                std::filesystem::path(original.repository) / model.tritonName, target);
    }
    auto invalid = tf32;
    invalid.id = "invalid_tf32_candidate";
    invalid.tritonName = "invalid_tf32_candidate";
    invalid.externalVersion = "invalid";
    invalid.tritonVersion = 1;
    std::filesystem::create_directories(repository.path / invalid.tritonName / "1");
    std::ofstream(repository.path / invalid.tritonName / "1/model.plan") << "invalid-engine";
    auto settings = original;
    settings.repository = repository.path.string();
    for (const auto batchSize : {1U, 2U, 3U, 4U, 8U})
    {
        SCOPED_TRACE("TF32 batch=" + std::to_string(batchSize));
        auto models = configured;
        for (auto& model : models)
            if (model.id == tf32.id && model.externalVersion == tf32.externalVersion)
            {
                model.batching.maxQueueDelayMicroseconds = 100000;
                model.batching.preferredBatchSizes = {batchSize == 8 ? 8U : 4U};
            }
        models.push_back(invalid);
        BackendService service(settings, models);
        service.selectModel({tf32.id, tf32.externalVersion});
        const auto generation = service.status().at("activeModel").at("generation");
        const auto before = settledStatistics(service, 1);
        std::promise<void> release;
        const auto barrier = release.get_future().share();
        std::vector<std::promise<void>> arrived(batchSize);
        std::vector<std::future<InferenceResult>> pending;
        for (std::size_t index = 0; index < batchSize; ++index)
            pending.push_back(std::async(std::launch::async, [&, index]
            {
                arrived[index].set_value();
                barrier.wait();
                return service.infer(request, tf32.id, tf32.externalVersion,
                    "tf32-" + std::to_string(batchSize) + "-" + std::to_string(index));
            }));
        for (auto& arrival : arrived) arrival.get_future().wait();
        release.set_value();
        for (auto& pendingResult : pending)
        {
            const auto result = pendingResult.get();
            expectValid(result, image, tf32);
            expectTf32Reference(result, references, batchSize);
        }
        const auto after = settledStatistics(service, before.at("inference_count").get<std::uint64_t>() + batchSize);
        EXPECT_EQ(after.at("inference_count").get<std::uint64_t>() - before.at("inference_count").get<std::uint64_t>(), batchSize);
        EXPECT_EQ(after.at("execution_count").get<std::uint64_t>() - before.at("execution_count").get<std::uint64_t>(), 1U);
        EXPECT_EQ(executionsForBatch(after, batchSize) - executionsForBatch(before, batchSize), 1U);
        if (batchSize != 4) continue;
        EXPECT_THROW(service.selectModel({invalid.id, invalid.externalVersion}), std::runtime_error);
        const auto failed = service.status();
        EXPECT_EQ(failed.at("activeModel").at("generation"), generation);
        EXPECT_EQ(failed.at("activeModel").at("version"), tf32.externalVersion);
        EXPECT_TRUE(failed.at("modelReady"));
        EXPECT_FALSE(failed.at("switching"));
        EXPECT_TRUE(failed.at("pendingModel").is_null());
        EXPECT_FALSE(failed.at("lastError").get<std::string>().empty());
        expectTf32Reference(service.infer(request, tf32.id, tf32.externalVersion, "tf32-after-failure"), references, 1);
        ASSERT_TRUE(waitForRetirement(service));
        service.selectModel({"best", "2.0.0"});
        EXPECT_EQ(service.status().at("activeModel").at("version"), "2.0.0");
        expectBaseline(service.infer(request, "best", "2.0.0", "back-to-fp32"), "best");
        EXPECT_TRUE(waitForRetirement(service));
    }
}

} // namespace
