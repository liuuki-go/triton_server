#include "api/BackendHttpServer.h"
#include "api/MetadataCodec.h"

#include <gtest/gtest.h>
#include <opencv2/imgcodecs.hpp>

#include <atomic>
#include <memory>
#include <mutex>
#include <thread>

using namespace triton_backend;

namespace
{

using Json = nlohmann::json;

class MockBackendService final : public IBackendService
{
public:
    void selectModel(const ModelSelection& selection) override
    {
        if (selection.modelId != "best") throw InvalidInferenceRequest("MODEL_NOT_FOUND", "未注册模型");
        if (selection.version != "1.0.0" && selection.version != "2.0.0")
            throw InvalidInferenceRequest("MODEL_VERSION_NOT_FOUND", "未注册版本");
        if (selectionFailure == 1) throw BackendNotReady("模型加载尚未完成");
        if (selectionFailure == 2) throw std::runtime_error("候选模型加载失败");
        ++selectCalls;
        std::lock_guard<std::mutex> lock(mutex);
        selected = selection;
        ready = true;
    }
    Json status() const override
    {
        const bool current = ready;
        std::lock_guard<std::mutex> lock(mutex);
        return {{"ok", true}, {"serviceAlive", true}, {"state", current ? "running" : "stopped"},
            {"modelReady", current}, {"activeModel", current ? Json{{"id", selected.modelId},
                {"version", selected.version}, {"tritonName", "best"}, {"tritonVersion", 1},
                {"inputs", Json::array({"images"})}} : Json(nullptr)},
            {"apiVersion", {{"major", 1}, {"minor", 0}}}, {"triton", {{"name", "mock"}}},
            {"switching", switching}, {"pendingModel", pendingModel}, {"lastError", lastError},
            {"retiringModels", Json::array()}, {"inFlight", 3},
            {"models", Json::array({Json{{"id", "best"}, {"displayName", "Best model"},
                {"versions", Json::array({
                    Json{{"version", "1.0.0"}, {"available", true}, {"ready", current},
                        {"tritonVersion", 1}, {"inputSize", 960}},
                    Json{{"version", "2.0.0"}, {"available", false}, {"ready", false},
                        {"tritonVersion", 2}, {"batching", {{"maxBatchSize", 8}}}}})}}})}};
    }
    InferenceResult infer(const InferenceRequest& request, const std::string& modelId,
                          const std::string& version, const std::string& requestId) override
    {
        ++inferCalls;
        {
            std::lock_guard<std::mutex> lock(mutex);
            lastRequest = request;
            lastModelId = modelId;
            lastVersion = version;
            lastRequestId = requestId;
        }
        if (failure == 1) throw BackendNotReady("模型尚未就绪");
        if (failure == 2) throw InferenceOverloaded("请求已满");
        if (failure == 3) throw std::runtime_error(std::string("invalid byte: ") + '\xFF');
        if (failure == 4) throw InvalidInferenceRequest("MODEL_NOT_ACTIVE", "已切换到其他版本");
        InferenceResult result;
        result.confidenceThreshold = request.confidenceThreshold;
        result.detections.push_back(Detection{0, 0.8F, 0, 0, 1, 1, "测试类别"});
        return result;
    }

    std::atomic<bool> ready{false};
    std::atomic<int> selectCalls{0};
    std::atomic<int> selectionFailure{0};
    std::atomic<int> inferCalls{0};
    std::atomic<int> failure{0};
    mutable std::mutex mutex;
    ModelSelection selected{"best", "1.0.0"};
    bool switching{false};
    Json pendingModel = nullptr;
    std::string lastError;
    InferenceRequest lastRequest;
    std::string lastModelId;
    std::string lastVersion;
    std::string lastRequestId;
};

class BackendHttpServerTest : public ::testing::Test
{
protected:
    void SetUp() override
    {
        BackendConfig config;
        config.server.port = 0;
        config.server.maxRequestBodyBytes = 65536;
        config.server.allowedOrigins = {"http://localhost:8090"};
        config.models.push_back(ModelConfig{"best", "1.0.0", "best", 1, 960, {{0, "测试类别"}}});
        config.models.push_back(ModelConfig{"best", "2.0.0", "best", 2, 960, {{0, "测试类别"}}});
        server = std::make_unique<BackendHttpServer>(service, config);
        const int port = server->bind();
        ASSERT_GT(port, 0);
        listener = std::thread([this] { server->listenAfterBind(); });
        client = std::make_unique<httplib::Client>("127.0.0.1", port);
        client->set_connection_timeout(2);
        client->set_read_timeout(2);
    }
    void TearDown() override
    {
        if (server) server->stop();
        if (listener.joinable()) listener.join();
    }
    Json metadata() const
    {
        return {{"SchemaVersion", "1.0"}, {"RequestId", "mock-frame-42"},
            {"Sequence", 42}, {"FrameId", 42}, {"CameraId", "camera-1"}, {"ChannelId", "1"},
            {"CapturedAtUtc", "2026-10-14T01:02:03Z"}, {"Width", 2}, {"Height", 2},
            {"SourcePixelFormat", "BGR8Packed"}, {"ImageEncoding", "Raw"},
            {"ModelId", "best"}, {"RequestedModelVersion", "1.0.0"},
            {"PixelFormat", "BGR8Packed"}, {"StrideBytes", 6}, {"RowOrder", "TopDown"},
            {"ConfidenceThreshold", 0.4}};
    }
    httplib::Result infer(const Json& body, const std::string& image = std::string(12, '\x12'),
                          const std::string& contentType = "application/octet-stream")
    {
        const httplib::MultipartFormDataItems parts{
            {"metadata", body.dump(), "", "application/json; charset=utf-8"},
            {"image", image, "", contentType}};
        return client->Post("/infer", parts);
    }
    void expectSelectionError(const httplib::Result& response, int status, const std::string& error,
                              bool modelReady = false, const Json& activeModel = nullptr)
    {
        ASSERT_TRUE(response);
        ASSERT_EQ(response->status, status);
        const auto body = Json::parse(response->body);
        ASSERT_TRUE(body.at("message").is_string());
        EXPECT_FALSE(body.at("message").get<std::string>().empty());
        EXPECT_EQ(body, (Json{{"ok", false}, {"modelReady", modelReady}, {"activeModel", activeModel},
            {"error", error}, {"message", body.at("message")}}));
    }

    MockBackendService service;
    std::unique_ptr<BackendHttpServer> server;
    std::unique_ptr<httplib::Client> client;
    std::thread listener;
};

TEST_F(BackendHttpServerTest, StatusCombinesReadinessCatalogAndCapabilitiesWithoutLoadingModels)
{
    const auto initial = client->Get("/api/status");
    ASSERT_TRUE(initial);
    ASSERT_EQ(initial->status, 200);
    const auto body = Json::parse(initial->body);
    EXPECT_TRUE(body.at("serviceAlive"));
    EXPECT_FALSE(body.at("modelReady"));
    EXPECT_TRUE(body.at("activeModel").is_null());
    EXPECT_EQ(body.at("triton").at("name"), "mock");
    EXPECT_EQ(body.at("apiVersion").at("major"), 1);
    EXPECT_EQ(body.at("schemaVersion"), "1.0");
    EXPECT_EQ(body.at("imageEncodings"), Json::array({"Png", "Raw"}));
    EXPECT_EQ(body.at("pixelFormats"), Json::array({"Mono8", "RGB8Packed", "BGR8Packed"}));
    EXPECT_EQ(body.at("heartbeatIntervalMs"), 2000);
    EXPECT_GE(body.at("uptimeMs").get<std::int64_t>(), 0);
    EXPECT_FALSE(body.at("switching"));
    EXPECT_TRUE(body.at("pendingModel").is_null());
    EXPECT_TRUE(body.at("retiringModels").empty());
    EXPECT_EQ(body.at("models")[0].at("versions").size(), 2U);
    EXPECT_FALSE(body.contains("health"));
    EXPECT_EQ(service.selectCalls, 0);
}

TEST_F(BackendHttpServerTest, SimpleStatusReturnsOnlyHostFieldsAndKeepsUnloadedModelsSelectable)
{
    const Json expected{{"ok", true}, {"state", "stopped"}, {"modelReady", false},
        {"activeModel", nullptr}, {"switching", false}, {"pendingModel", nullptr},
        {"lastError", ""}, {"schemaVersion", "1.0"},
        {"imageEncodings", Json::array({"Png", "Raw"})}, {"heartbeatIntervalMs", 2000},
        {"models", Json::array({Json{{"id", "best"}, {"versions", Json::array({
            Json{{"version", "1.0.0"}, {"available", true}},
            Json{{"version", "2.0.0"}, {"available", false}}})}}})}};
    for (const char* query : {"?simple", "?simple=", "?simple=true", "?simple=1",
        "?simple=true&simple=true", "?simple&simple", "?simple=true&%73imple=true"})
    {
        SCOPED_TRACE(query);
        const auto response = client->Get(std::string("/api/status") + query);
        ASSERT_TRUE(response);
        ASSERT_EQ(response->status, 200);
        EXPECT_EQ(Json::parse(response->body), expected);
    }
    EXPECT_EQ(service.selectCalls, 0);
}

TEST_F(BackendHttpServerTest, SimpleStatusReducesActiveAndPendingModelsToTheirIdentities)
{
    {
        std::lock_guard<std::mutex> lock(service.mutex);
        service.ready = true;
        service.switching = true;
        service.pendingModel = {{"id", "best"}, {"version", "2.0.0"},
            {"tritonVersion", 2}, {"generation", 7}, {"inputs", Json::array({"images"})}};
    }
    const auto response = client->Get("/api/status?simple=true");
    ASSERT_TRUE(response);
    ASSERT_EQ(response->status, 200);
    const auto body = Json::parse(response->body);
    EXPECT_EQ(body.size(), 11U);
    EXPECT_EQ(body.at("state"), "running");
    EXPECT_TRUE(body.at("modelReady"));
    EXPECT_TRUE(body.at("switching"));
    EXPECT_EQ(body.at("activeModel"), (Json{{"id", "best"}, {"version", "1.0.0"}}));
    EXPECT_EQ(body.at("pendingModel"), (Json{{"id", "best"}, {"version", "2.0.0"}}));
    EXPECT_FALSE(body.contains("inFlight"));
    EXPECT_FALSE(body.contains("triton"));
    EXPECT_FALSE(body.contains("retiringModels"));
}

TEST_F(BackendHttpServerTest, SimpleStatusPreservesLoadErrorAndTheStillReadyActiveModel)
{
    {
        std::lock_guard<std::mutex> lock(service.mutex);
        service.ready = true;
        service.lastError = "候选模型加载失败";
    }
    const auto response = client->Get("/api/status?simple=1");
    ASSERT_TRUE(response);
    ASSERT_EQ(response->status, 200);
    const auto body = Json::parse(response->body);
    EXPECT_TRUE(body.at("ok"));
    EXPECT_TRUE(body.at("modelReady"));
    EXPECT_EQ(body.at("activeModel"), (Json{{"id", "best"}, {"version", "1.0.0"}}));
    EXPECT_EQ(body.at("lastError"), "候选模型加载失败");
    EXPECT_TRUE(body.at("pendingModel").is_null());
}

TEST_F(BackendHttpServerTest, FullStatusRemainsTheDefaultAndCanBeExplicitlyRequested)
{
    const auto response = client->Get("/api/status");
    ASSERT_TRUE(response);
    ASSERT_EQ(response->status, 200);
    auto expected = Json::parse(response->body);
    ASSERT_TRUE(expected.contains("uptimeMs"));
    ASSERT_TRUE(expected.contains("pixelFormats"));
    ASSERT_TRUE(expected.contains("triton"));
    ASSERT_TRUE(expected.contains("apiVersion"));
    ASSERT_TRUE(expected.at("models")[0].at("versions")[0].contains("tritonVersion"));
    expected.erase("uptimeMs");
    for (const char* query : {"?simple=false", "?simple=0"})
    {
        SCOPED_TRACE(query);
        const auto full = client->Get(std::string("/api/status") + query);
        ASSERT_TRUE(full);
        ASSERT_EQ(full->status, 200);
        auto body = Json::parse(full->body);
        ASSERT_TRUE(body.contains("uptimeMs"));
        body.erase("uptimeMs");
        EXPECT_EQ(body, expected);
    }
}

TEST_F(BackendHttpServerTest, StatusRejectsInvalidOrConflictingSimpleParametersWithoutLoadingModels)
{
    for (const char* query : {"?simple=yes", "?simple=2", "?simple=TRUE", "?simple=False",
        "?simple=0&simple=1", "?simple&simple=false", "?simple=true&simple=1"})
    {
        SCOPED_TRACE(query);
        const auto response = client->Get(std::string("/api/status") + query);
        ASSERT_TRUE(response);
        ASSERT_EQ(response->status, 400);
        const auto body = Json::parse(response->body);
        EXPECT_FALSE(body.at("ok"));
        EXPECT_EQ(body.at("error"), "INVALID_QUERY");
        EXPECT_TRUE(body.at("message").is_string());
        EXPECT_FALSE(body.at("message").get<std::string>().empty());
    }
    EXPECT_EQ(service.selectCalls, 0);
}

TEST_F(BackendHttpServerTest, RemovedRoutesReturnNotFoundAndCannotControlModels)
{
    for (const char* route : {"/health", "/api/health", "/api/info", "/api/models"})
    {
        const auto response = client->Get(route);
        ASSERT_TRUE(response);
        EXPECT_EQ(response->status, 404);
        EXPECT_EQ(Json::parse(response->body).at("error"), "NOT_FOUND");
    }
    for (const char* route : {"/api/start", "/api/stop"})
    {
        const auto response = client->Post(route, "{}", "application/json");
        ASSERT_TRUE(response);
        EXPECT_EQ(response->status, 404);
        EXPECT_EQ(Json::parse(response->body).at("error"), "NOT_FOUND");
    }
    EXPECT_EQ(service.selectCalls, 0);
}

TEST_F(BackendHttpServerTest, OneSelectionRouteLoadsInitiallyAndSwitchesToAnotherVersion)
{
    for (const char* version : {"1.0.0", "2.0.0", "2.0.0"})
    {
        const Json selection{{"ModelId", "best"}, {"RequestedModelVersion", version}};
        const auto selected = client->Post("/api/models/select", selection.dump(), "application/json");
        ASSERT_TRUE(selected);
        ASSERT_EQ(selected->status, 200);
        EXPECT_EQ(Json::parse(selected->body), (Json{{"ok", true}, {"modelReady", true},
            {"activeModel", {{"id", "best"}, {"version", version}}}}));
    }
    const auto status = client->Get("/api/status");
    ASSERT_TRUE(status);
    ASSERT_EQ(status->status, 200);
    const auto full = Json::parse(status->body);
    EXPECT_EQ(full.at("activeModel").at("version"), "2.0.0");
    EXPECT_TRUE(full.contains("serviceAlive"));
    EXPECT_TRUE(full.contains("triton"));
    EXPECT_TRUE(full.contains("pixelFormats"));
    EXPECT_TRUE(full.at("activeModel").contains("tritonVersion"));
    EXPECT_EQ(full.at("models")[0].at("versions").size(), 2U);
    auto frame = metadata();
    frame["RequestedModelVersion"] = "2.0.0";
    const auto response = infer(frame);
    ASSERT_TRUE(response);
    EXPECT_EQ(response->status, 200);
    EXPECT_EQ(service.lastVersion, "2.0.0");
}

TEST_F(BackendHttpServerTest, MissingSelectionReturnsClearErrorAndMalformedControlsDoNotSwitch)
{
    for (const Json& body : {Json::object(), Json{{"ModelId", "best"}},
        Json{{"ModelId", "best"}, {"RequestedModelVersion", 2}},
        Json{{"ModelId", "best"}, {"RequestedModelVersion", "1.0.0"}, {"extra", true}}})
    {
        const auto response = client->Post("/api/models/select", body.dump(), "application/json");
        expectSelectionError(response, 400,
            body.contains("RequestedModelVersion") && body.at("RequestedModelVersion").is_number()
                ? "INVALID_METADATA" : "INVALID_REQUEST");
    }
    for (const auto& missing : {ModelSelection{"unknown", "1.0.0"}, ModelSelection{"best", "unknown"}})
    {
        const Json body{{"ModelId", missing.modelId}, {"RequestedModelVersion", missing.version}};
        const auto response = client->Post("/api/models/select", body.dump(), "application/json");
        expectSelectionError(response, 404,
            missing.modelId == "unknown" ? "MODEL_NOT_FOUND" : "MODEL_VERSION_NOT_FOUND");
    }
    EXPECT_EQ(service.selectCalls, 0);
}

TEST_F(BackendHttpServerTest, SelectionFailuresWithoutAnActiveModelReturnOnlyErrorAndNullIdentity)
{
    const Json selection{{"ModelId", "best"}, {"RequestedModelVersion", "1.0.0"}};
    const auto malformed = client->Post("/api/models/select", "not-json", "application/json");
    expectSelectionError(malformed, 400, "INVALID_JSON");
    const auto unsupported = client->Post("/api/models/select", selection.dump(), "text/plain");
    expectSelectionError(unsupported, 415, "UNSUPPORTED_CONTENT_TYPE");
    const auto oversized = client->Post("/api/models/select", std::string(20000, 'x'), "application/json");
    expectSelectionError(oversized, 413, "METADATA_TOO_LARGE");
    const auto payloadTooLarge = client->Post("/api/models/select", std::string(65537, 'x'), "application/json");
    expectSelectionError(payloadTooLarge, 413, "PAYLOAD_TOO_LARGE");
    for (int failure : {1, 2})
    {
        service.selectionFailure = failure;
        const auto response = client->Post("/api/models/select", selection.dump(), "application/json");
        expectSelectionError(response, failure == 1 ? 503 : 500,
            failure == 1 ? "BACKEND_NOT_READY" : "MODEL_CONTROL_FAILED");
    }
    EXPECT_EQ(service.selectCalls, 0);
}

TEST_F(BackendHttpServerTest, SelectionRejectsInvalidContentAndPreservesActiveModelOnLoadFailure)
{
    const Json selection{{"ModelId", "best"}, {"RequestedModelVersion", "1.0.0"}};
    const auto selected = client->Post("/api/models/select", selection.dump(), "application/json");
    ASSERT_TRUE(selected);
    ASSERT_EQ(selected->status, 200);
    const Json activeModel{{"id", "best"}, {"version", "1.0.0"}};
    for (const char* invalid : {"{}", "[]", "null", "not-json"})
    {
        const auto response = client->Post("/api/models/select", invalid, "application/json");
        expectSelectionError(response, 400,
            std::string(invalid) == "not-json" ? "INVALID_JSON" : "INVALID_REQUEST", true, activeModel);
    }
    const auto unsupported = client->Post("/api/models/select", selection.dump(), "text/plain");
    expectSelectionError(unsupported, 415, "UNSUPPORTED_CONTENT_TYPE", true, activeModel);
    const auto oversized = client->Post("/api/models/select", std::string(20000, 'x'), "application/json");
    expectSelectionError(oversized, 413, "METADATA_TOO_LARGE", true, activeModel);
    const auto payloadTooLarge = client->Post("/api/models/select", std::string(65537, 'x'), "application/json");
    expectSelectionError(payloadTooLarge, 413, "PAYLOAD_TOO_LARGE", true, activeModel);
    for (const auto& missing : {ModelSelection{"unknown", "1.0.0"}, ModelSelection{"best", "unknown"}})
    {
        const Json target{{"ModelId", missing.modelId}, {"RequestedModelVersion", missing.version}};
        const auto response = client->Post("/api/models/select", target.dump(), "application/json");
        expectSelectionError(response, 404,
            missing.modelId == "unknown" ? "MODEL_NOT_FOUND" : "MODEL_VERSION_NOT_FOUND", true, activeModel);
    }
    for (int failure : {1, 2})
    {
        service.selectionFailure = failure;
        const Json target{{"ModelId", "best"}, {"RequestedModelVersion", "2.0.0"}};
        const auto response = client->Post("/api/models/select", target.dump(), "application/json");
        expectSelectionError(response, failure == 1 ? 503 : 500,
            failure == 1 ? "BACKEND_NOT_READY" : "MODEL_CONTROL_FAILED", true, activeModel);
    }
    EXPECT_EQ(service.selectCalls, 1);
}

TEST_F(BackendHttpServerTest, PreflightOnlyAcceptsTheThreePublicRoutes)
{
    const httplib::Headers headers{{"Origin", "http://localhost:8090"},
        {"Access-Control-Request-Method", "GET"}};
    const auto status = client->Options("/api/status", headers);
    ASSERT_TRUE(status);
    EXPECT_EQ(status->status, 204);
    for (const char* route : {"/api/models/select", "/infer"})
    {
        auto postHeaders = headers;
        postHeaders.erase("Access-Control-Request-Method");
        postHeaders.emplace("Access-Control-Request-Method", "POST");
        const auto response = client->Options(route, postHeaders);
        ASSERT_TRUE(response);
        EXPECT_EQ(response->status, 204);
    }
    for (const char* route : {"/health", "/api/health", "/api/info", "/api/models", "/api/start", "/api/stop"})
    {
        const auto response = client->Options(route, headers);
        ASSERT_TRUE(response);
        EXPECT_EQ(response->status, 403);
        EXPECT_EQ(Json::parse(response->body).at("error"), "INVALID_PREFLIGHT");
    }
}

TEST_F(BackendHttpServerTest, RawInferencePassesBusinessRequestAndEchoesAllowedMetadata)
{
    auto body = metadata();
    body["UnknownClientField"] = "do not echo";
    const auto response = infer(body);
    ASSERT_TRUE(response);
    ASSERT_EQ(response->status, 200);
    const auto result = Json::parse(response->body);
    EXPECT_EQ(result.at("RequestId"), "mock-frame-42");
    EXPECT_EQ(result.at("detectionCount"), 1);
    EXPECT_EQ(result.at("detections")[0].at("className"), "测试类别");
    EXPECT_FALSE(result.contains("UnknownClientField"));
    EXPECT_EQ(service.inferCalls, 1);
    std::lock_guard<std::mutex> lock(service.mutex);
    ASSERT_TRUE(service.lastRequest.image);
    EXPECT_EQ(service.lastRequest.image->width, 2);
    EXPECT_EQ(service.lastRequest.image->strideBytes, 6U);
    EXPECT_EQ(*service.lastRequest.image->data, std::string(12, '\x12'));
    EXPECT_FLOAT_EQ(service.lastRequest.confidenceThreshold, 0.4F);
    EXPECT_EQ(service.lastModelId, "best");
    EXPECT_EQ(service.lastVersion, "1.0.0");
    EXPECT_EQ(service.lastRequestId, "mock-frame-42");
}

TEST_F(BackendHttpServerTest, PngInferenceDoesNotRequireRawLayoutAndPreservesEncodedBytes)
{
    auto body = metadata();
    body["ImageEncoding"] = "Png";
    for (const char* field : {"PixelFormat", "StrideBytes", "RowOrder"}) body.erase(field);
    std::vector<unsigned char> encoded;
    ASSERT_TRUE(cv::imencode(".png", cv::Mat(2, 2, CV_8UC3, cv::Scalar(10, 20, 30)), encoded));
    const std::string bytes(encoded.begin(), encoded.end());
    for (const char* contentType : {"image/png", "application/octet-stream"})
    {
        const auto response = infer(body, bytes, contentType);
        ASSERT_TRUE(response);
        ASSERT_EQ(response->status, 200);
        const auto result = Json::parse(response->body);
        EXPECT_EQ(result.at("ImageEncoding"), "Png");
        EXPECT_EQ(result.at("RequestId"), body.at("RequestId"));
        EXPECT_FALSE(result.contains("StrideBytes"));
        std::lock_guard<std::mutex> lock(service.mutex);
        EXPECT_FALSE(service.lastRequest.image);
        ASSERT_TRUE(service.lastRequest.pngImage);
        EXPECT_EQ(service.lastRequest.pngImage->width, 2);
        EXPECT_EQ(service.lastRequest.pngImage->height, 2);
        EXPECT_EQ(*service.lastRequest.pngImage->data, bytes);
        EXPECT_FLOAT_EQ(service.lastRequest.confidenceThreshold, 0.4F);
    }
}

TEST_F(BackendHttpServerTest, InvalidPngHeaderOrSizeIsRejectedBeforeCallingService)
{
    auto body = metadata();
    body["ImageEncoding"] = "Png";
    const auto invalid = infer(body, "not-png", "image/png");
    ASSERT_TRUE(invalid);
    EXPECT_EQ(invalid->status, 400);
    EXPECT_EQ(Json::parse(invalid->body).at("error"), "INVALID_PNG_IMAGE");
    std::vector<unsigned char> encoded;
    ASSERT_TRUE(cv::imencode(".png", cv::Mat(2, 3, CV_8UC3, cv::Scalar(10, 20, 30)), encoded));
    const auto mismatch = infer(body, std::string(encoded.begin(), encoded.end()), "image/png");
    ASSERT_TRUE(mismatch);
    EXPECT_EQ(mismatch->status, 400);
    EXPECT_EQ(Json::parse(mismatch->body).at("error"), "INVALID_PNG_IMAGE");
    EXPECT_EQ(service.inferCalls, 0);
}

TEST_F(BackendHttpServerTest, RejectsInvalidModelBeforeCallingService)
{
    auto body = metadata();
    body["RequestedModelVersion"] = "missing-version";
    const auto response = infer(body);
    ASSERT_TRUE(response);
    EXPECT_EQ(response->status, 409);
    EXPECT_EQ(Json::parse(response->body).at("error"), "MODEL_VERSION_MISMATCH");
    EXPECT_EQ(service.inferCalls, 0);
}

TEST_F(BackendHttpServerTest, MapsServiceFailuresAndKeepsErrorJsonValidForInvalidUtf8)
{
    for (const auto failure : {1, 2, 3, 4})
    {
        service.failure = failure;
        const auto response = infer(metadata());
        ASSERT_TRUE(response);
        EXPECT_EQ(response->status, failure == 3 ? 500 : failure == 4 ? 409 : 503);
        const auto body = Json::parse(response->body);
        EXPECT_EQ(body.at("RequestId"), "mock-frame-42");
        const auto expected = failure == 1 ? "BACKEND_NOT_READY" : failure == 2 ? "SERVICE_BUSY" : failure == 4 ? "MODEL_NOT_ACTIVE" : "INFERENCE_FAILED";
        EXPECT_EQ(body.at("error"), expected);
    }
}

} // namespace
