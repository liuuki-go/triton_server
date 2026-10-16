// Standalone reference exporter; intentionally not a production CMake target.
#include "inference/ImagePipeline.h"

#include <NvInfer.h>
#include <NvInferPlugin.h>
#include <cuda_runtime_api.h>
#include <nlohmann/json.hpp>
#include <openssl/sha.h>

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace fs = std::filesystem;
using Json = nlohmann::json;

struct Logger final : nvinfer1::ILogger
{
    void log(Severity severity, const char* message) noexcept override
    {
        if (severity <= Severity::kWARNING) std::cerr << message << '\n';
    }
} logger;

void checkCuda(cudaError_t error)
{
    if (error != cudaSuccess) throw std::runtime_error(cudaGetErrorString(error));
}

std::string readFile(const fs::path& path)
{
    std::ifstream input(path, std::ios::binary);
    if (!input) throw std::runtime_error("Cannot read " + path.string());
    return {std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>()};
}

std::string sha256(const std::string& bytes)
{
    unsigned char digest[SHA256_DIGEST_LENGTH];
    if (SHA256(reinterpret_cast<const unsigned char*>(bytes.data()), bytes.size(), digest) == nullptr)
        throw std::runtime_error("Cannot compute SHA256");
    std::ostringstream result;
    for (const auto value : digest) result << std::hex << std::setw(2) << std::setfill('0') << static_cast<int>(value);
    return result.str();
}

struct Runner
{
    std::unique_ptr<nvinfer1::IRuntime> runtime{nvinfer1::createInferRuntime(logger)};
    std::unique_ptr<nvinfer1::ICudaEngine> engine;
    std::unique_ptr<nvinfer1::IExecutionContext> context;
    cudaStream_t stream{};
    float* input{};
    float* output{};

    explicit Runner(const std::string& serialized)
    {
        if (!runtime) throw std::runtime_error("Cannot create TensorRT runtime");
        engine.reset(runtime->deserializeCudaEngine(serialized.data(), serialized.size()));
        if (!engine) throw std::runtime_error("Engine deserialization failed");
        if (engine->getNbIOTensors() != 2 || engine->getNbOptimizationProfiles() != 1 ||
            engine->getTensorDataType("images") != nvinfer1::DataType::kFLOAT ||
            engine->getTensorDataType("output0") != nvinfer1::DataType::kFLOAT)
            throw std::runtime_error("Expected single-profile FP32 images and output0");
        const auto minimum = engine->getProfileShape("images", 0, nvinfer1::OptProfileSelector::kMIN);
        const auto maximum = engine->getProfileShape("images", 0, nvinfer1::OptProfileSelector::kMAX);
        if (minimum.nbDims != 4 || maximum.nbDims != 4 || minimum.d[0] != 1 || maximum.d[0] != 8 ||
            minimum.d[1] != 3 || maximum.d[1] != 3 || minimum.d[2] != 960 || maximum.d[2] != 960 ||
            minimum.d[3] != 960 || maximum.d[3] != 960)
            throw std::runtime_error("Expected batch profile [1..8,3,960,960]");
        context.reset(engine->createExecutionContext());
        if (!context) throw std::runtime_error("Cannot create context");
        checkCuda(cudaMalloc(reinterpret_cast<void**>(&input), 8ULL * 3 * 960 * 960 * sizeof(float)));
        checkCuda(cudaMalloc(reinterpret_cast<void**>(&output), 8ULL * 300 * 6 * sizeof(float)));
        checkCuda(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
        if (!context->setTensorAddress("images", input) || !context->setTensorAddress("output0", output))
            throw std::runtime_error("Cannot bind tensors");
    }

    ~Runner()
    {
        if (stream) cudaStreamSynchronize(stream);
        if (input) cudaFree(input);
        if (output) cudaFree(output);
        if (stream) cudaStreamDestroy(stream);
    }

    std::vector<float> infer(const triton_backend::PreparedImage& image, int batch)
    {
        if (!context->setInputShape("images", nvinfer1::Dims4{batch, 3, 960, 960}))
            throw std::runtime_error("Input shape rejected");
        const auto shape = context->getTensorShape("output0");
        if (shape.nbDims != 3 || shape.d[0] != batch || shape.d[1] != 300 || shape.d[2] != 6)
            throw std::runtime_error("Output shape is not [N,300,6]");
        std::vector<float> hostInput;
        hostInput.reserve(batch * image.tensor.size());
        for (int position = 0; position < batch; ++position)
            hostInput.insert(hostInput.end(), image.tensor.begin(), image.tensor.end());
        std::vector<float> hostOutput(batch * 1800);
        checkCuda(cudaMemcpyAsync(input, hostInput.data(), hostInput.size() * sizeof(float), cudaMemcpyHostToDevice, stream));
        if (!context->enqueueV3(stream)) throw std::runtime_error("enqueueV3 failed");
        checkCuda(cudaMemcpyAsync(hostOutput.data(), output, hostOutput.size() * sizeof(float), cudaMemcpyDeviceToHost, stream));
        checkCuda(cudaStreamSynchronize(stream));
        if (std::any_of(hostOutput.begin(), hostOutput.end(), [](float value) { return !std::isfinite(value); }))
            throw std::runtime_error("Engine output contains NaN or Inf");
        return hostOutput;
    }
};

int main(int argc, char** argv)
{
    if (argc != 3 && argc != 4)
    {
        std::cerr << "Usage: export_tf32_reference ENGINE OUTPUT_JSON [PROJECT_ROOT]\n";
        return 1;
    }
    try
    {
        const fs::path project = argc == 4 ? argv[3] : "/opt/triton_back";
        const fs::path outputPath = argv[2];
        if (fs::exists(outputPath)) throw std::runtime_error("Output already exists; refusing to replace a reference");
        const auto serialized = readFile(argv[1]);
        const auto engineSha256 = sha256(serialized);
        const auto manifest = Json::parse(readFile(project / "models/manifest.json"));
        const auto manifestModel = std::find_if(manifest.at("models").begin(), manifest.at("models").end(),
            [](const auto& item) { return item.at("id") == "best" && item.at("tritonVersion") == 3; });
        if (manifestModel == manifest.at("models").end()) throw std::runtime_error("Missing best version 3 model manifest entry");
        const auto manifestPath = project / manifestModel->at("enginePath").get<std::string>();
        if (fs::canonical(argv[1]) != fs::canonical(manifestPath) || engineSha256 != manifestModel->at("sha256"))
            throw std::runtime_error("The Engine path or SHA256 differs from the best version 3 model manifest");
        const auto fixtures = Json::parse(readFile(project / "tests/fixtures/fixtures.json"));
        const auto foundFixture = std::find_if(fixtures.at("fixtures").begin(), fixtures.at("fixtures").end(),
            [](const auto& item) { return item.at("modelId") == "best"; });
        if (foundFixture == fixtures.at("fixtures").end()) throw std::runtime_error("Missing best fixture");
        const auto& fixture = *foundFixture;
        if (fixture.at("pixelFormat") != "BGR8Packed" || fixture.at("rowOrder") != "TopDown")
            throw std::runtime_error("Expected BGR8Packed TopDown fixture");
        const auto raw = readFile(project / "tests/fixtures" / fixture.at("rawFile").get<std::string>());
        const auto rawSha256 = sha256(raw);
        if (rawSha256 != fixture.at("sha256")) throw std::runtime_error("Fixture SHA256 mismatch");
        const auto configurations = Json::parse(readFile(project / "config/models.json"));
        const auto foundModel = std::find_if(configurations.at("models").begin(), configurations.at("models").end(),
            [](const auto& item) { return item.at("id") == "best" && item.at("businessVersion") == "3.0.0"; });
        if (foundModel == configurations.at("models").end()) throw std::runtime_error("Missing best/3.0.0 registration");
        const auto& model = *foundModel;
        if (model.at("tritonVersion") != 3 || model.at("inputSize") != 960 || model.at("batching").at("maxBatchSize") != 8)
            throw std::runtime_error("Unexpected best/3.0.0 model contract");
        const RawImage image{fixture.at("width"), fixture.at("height"), PixelFormat::BGR8Packed,
            fixture.at("strideBytes"), false, std::make_shared<const std::string>(raw)};
        validateRawImage(image);
        const auto prepared = triton_backend::prepareImage(image, 960);
        if (!initLibNvInferPlugins(&logger, "")) throw std::runtime_error("Cannot initialize TensorRT plugins");
        Runner runner(serialized);
        Json references = Json::array();
        for (int batch = 1; batch <= 8; ++batch)
        {
            const auto values = runner.infer(prepared, batch);
            for (int position = 0; position < batch; ++position)
            {
                const std::vector<float> output(values.begin() + position * 1800, values.begin() + (position + 1) * 1800);
                const auto detections = triton_backend::decodeDetections(output, prepared.letterbox, 0.25F);
                if (detections.empty()) throw std::runtime_error("The reference fixture must produce detections");
                Json items = Json::array();
                for (const auto& detection : detections)
                    items.push_back({{"classId", detection.classId},
                        {"className", model.at("classNames").at(std::to_string(detection.classId))},
                        {"confidence", detection.confidence}, {"box", {{"x1", detection.x1}, {"y1", detection.y1},
                            {"x2", detection.x2}, {"y2", detection.y2}}}});
                references.push_back({{"batchSize", batch}, {"batchPosition", position},
                    {"detectionCount", detections.size()}, {"detections", items}});
            }
        }
        const Json reference{{"reference", "Direct TensorRT best/3.0.0 TF32 Engine; identical best.raw in every batch position"},
            {"modelId", "best"}, {"businessVersion", "3.0.0"}, {"tritonVersion", 3},
            {"engineSha256", engineSha256}, {"rawSha256", rawSha256}, {"width", image.width}, {"height", image.height},
            {"inputSize", 960}, {"confidenceThreshold", 0.25}, {"confidenceTolerance", 0.00001},
            {"coordinateTolerance", 0.001}, {"references", references}};
        if (!outputPath.parent_path().empty()) fs::create_directories(outputPath.parent_path());
        std::ofstream result(outputPath);
        if (!result || !(result << reference.dump(2) << '\n')) throw std::runtime_error("Cannot write reference JSON");
        std::cout << "Exported " << references.size() << " direct TensorRT batch/position references\n";
        return 0;
    }
    catch (const std::exception& error)
    {
        std::cerr << "Reference export failed: " << error.what() << '\n';
        return 1;
    }
}
