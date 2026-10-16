#include "api/BackendHttpServer.h"
#include "api/BoundedHttpTaskQueue.h"
#include "api/MetadataCodec.h"

#include <algorithm>
#include <cmath>
#include <exception>
#include <stdexcept>

namespace triton_backend
{

using Json = nlohmann::json;

BackendHttpServer::BackendHttpServer(IBackendService& service, const BackendConfig& config)
    : service_(service), config_(config)
{
    server_.new_task_queue = [options = config.server]
    {
        return new BoundedHttpTaskQueue(options.workerCount, options.queueCapacity);
    };
    server_.set_payload_max_length(config.server.maxRequestBodyBytes);
    server_.set_read_timeout(config.server.readTimeoutSeconds, 0);
    server_.set_write_timeout(config.server.writeTimeoutSeconds, 0);
    server_.set_tcp_nodelay(true);
    registerRoutes();
}

int BackendHttpServer::bind()
{
    if (config_.server.port == 0) return server_.bind_to_any_port(config_.server.host);
    return server_.bind_to_port(config_.server.host, config_.server.port) ? config_.server.port : -1;
}

bool BackendHttpServer::listenAfterBind() { return server_.listen_after_bind(); }
void BackendHttpServer::stop() { server_.stop(); }

Json BackendHttpServer::status() const
{
    auto body = service_.status();
    body["schemaVersion"] = config_.server.schemaVersion;
    body["uptimeMs"] = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now() - startedAt_).count();
    body["heartbeatIntervalMs"] = 2000;
    body["pixelFormats"] = {"Mono8", "RGB8Packed", "BGR8Packed"};
    body["imageEncodings"] = {"Png", "Raw"};
    return body;
}

void BackendHttpServer::registerRoutes()
{
    server_.set_pre_routing_handler([this](const httplib::Request& request, httplib::Response& response)
    {
        if (request.has_header("Origin"))
        {
            const auto origin = request.get_header_value("Origin");
            const auto& allowed = config_.server.allowedOrigins;
            if (request.get_header_value_count("Origin") != 1 ||
                std::find(allowed.begin(), allowed.end(), origin) == allowed.end())
            {
                sendJson(response, 403, makeError("INVALID_ORIGIN", "Origin 不在允许列表中"));
                return httplib::Server::HandlerResponse::Handled;
            }
            response.set_header("Access-Control-Allow-Origin", origin);
            response.set_header("Vary", "Origin");
        }
        return httplib::Server::HandlerResponse::Unhandled;
    });
    server_.Options("/.*", [](const httplib::Request& request, httplib::Response& response)
    {
        const auto path = request.path;
        const bool query = path == "/api/status";
        const bool mutate = path == "/infer" || path == "/api/models/select";
        const auto method = request.get_header_value("Access-Control-Request-Method");
        if (!request.has_header("Origin") || (!query && !mutate) ||
            (query && method != "GET") || (mutate && method != "POST"))
        {
            sendJson(response, 403, makeError("INVALID_PREFLIGHT", "预检路径或方法无效"));
            return;
        }
        const auto headers = request.get_header_value("Access-Control-Request-Headers");
        if (!headers.empty() && !hasMediaType(headers, "content-type"))
        {
            sendJson(response, 403, makeError("INVALID_PREFLIGHT", "只允许 Content-Type 请求头"));
            return;
        }
        response.set_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS");
        response.set_header("Access-Control-Allow-Headers", "Content-Type");
        response.status = 204;
    });
    server_.Get("/api/status", [this](const httplib::Request&, httplib::Response& response)
    {
        sendJson(response, 200, status());
    });
    server_.Post("/api/models/select", [this](const httplib::Request& request, httplib::Response& response)
    {
        if (!hasMediaType(request.get_header_value("Content-Type"), "application/json", true))
        {
            sendJson(response, 415, makeError("UNSUPPORTED_CONTENT_TYPE", "控制请求使用 application/json"));
            return;
        }
        if (request.body.size() > config_.server.maxMetadataBytes)
        {
            sendJson(response, 413, makeError("METADATA_TOO_LARGE", "控制请求超过大小上限"));
            return;
        }
        Json body;
        ModelSelection selection;
        try
        {
            body = Json::parse(request.body);
            selection = parseModelSelection(body);
        }
        catch (const Json::exception& error)
        {
            sendJson(response, 400, makeError("INVALID_JSON", error.what()));
            return;
        }
        catch (const InvalidInferenceRequest& error)
        {
            sendJson(response, 400, makeError(error.code(), error.what()));
            return;
        }
        try
        {
            service_.selectModel(selection);
            sendJson(response, 200, status());
        }
        catch (const InvalidInferenceRequest& error)
        {
            auto result = status();
            result.update(makeError(error.code(), error.what()));
            sendJson(response, 404, result);
        }
        catch (const BackendNotReady& error)
        {
            auto result = status();
            result.update(makeError("BACKEND_NOT_READY", error.what()));
            sendJson(response, 503, result);
        }
        catch (const std::exception& error)
        {
            auto body = status();
            body["ok"] = false;
            body["error"] = "MODEL_CONTROL_FAILED";
            body["message"] = error.what();
            sendJson(response, 500, body);
        }
    });
    server_.Post("/infer", [this](const httplib::Request& request, httplib::Response& response)
    {
        Json metadata;
        const auto fail = [&metadata, &response](int code, const std::string& error, const std::string& message)
        {
            auto body = makeError(error, message);
            copyMetadata(metadata, body);
            sendJson(response, code, body);
        };
        if (!request.is_multipart_form_data())
        {
            fail(415, "UNSUPPORTED_CONTENT_TYPE", "请使用 multipart/form-data，包含 metadata 与 image");
            return;
        }
        if (request.get_header_value_count("Content-Length") != 1 || request.has_header("Transfer-Encoding"))
        {
            fail(400, "INVALID_CONTENT_LENGTH", "必须声明唯一 Content-Length，不使用 chunked");
            return;
        }
        if (request.files.count("metadata") != 1 || request.files.count("image") != 1 || request.files.size() != 2)
        {
            fail(400, "INVALID_MULTIPART", "必须且只能提供一个 metadata 和一个 image 部分");
            return;
        }
        const auto& metadataPart = request.files.find("metadata")->second;
        const auto& imagePart = request.files.find("image")->second;
        if (metadataPart.content.size() > config_.server.maxMetadataBytes)
        {
            fail(413, "METADATA_TOO_LARGE", "metadata 超过大小上限");
            return;
        }
        try { metadata = Json::parse(metadataPart.content); }
        catch (const Json::exception& error) { fail(400, "INVALID_JSON", error.what()); return; }
        if (!metadata.is_object()) { fail(400, "INVALID_METADATA", "metadata 必须为 JSON 对象"); return; }
        const bool png = metadata.contains("ImageEncoding") && metadata["ImageEncoding"] == "Png";
        if (!hasMediaType(metadataPart.content_type, "application/json", true) ||
            (!hasMediaType(imagePart.content_type, "application/octet-stream") &&
             !(png && hasMediaType(imagePart.content_type, "image/png"))))
        {
            fail(415, "UNSUPPORTED_CONTENT_TYPE", "metadata 为 application/json；PNG 的 image 可用 image/png 或 application/octet-stream");
            return;
        }
        try
        {
            InferenceRequest inference = parseInferenceImage(metadata, imagePart.content);
            if (metadata["SchemaVersion"] != config_.server.schemaVersion)
            {
                fail(400, "UNSUPPORTED_SCHEMA_VERSION", "只支持协议 1.0");
                return;
            }
            const auto modelId = metadata["ModelId"].get<std::string>();
            const auto version = metadata["RequestedModelVersion"].get<std::string>();
            const auto model = std::find_if(config_.models.begin(), config_.models.end(),
                [&modelId](const auto& value) { return value.id == modelId; });
            if (model == config_.models.end()) { fail(409, "MODEL_MISMATCH", "未配置该模型"); return; }
            if (std::none_of(config_.models.begin(), config_.models.end(), [&](const auto& value)
                { return value.id == modelId && value.externalVersion == version; }))
            { fail(409, "MODEL_VERSION_MISMATCH", "未注册该模型版本"); return; }
            if (metadata.contains("ConfidenceThreshold"))
            {
                if (!metadata["ConfidenceThreshold"].is_number())
                    throw InvalidInferenceRequest("INVALID_CONFIDENCE_THRESHOLD", "阈值必须是数字");
                const double threshold = metadata["ConfidenceThreshold"].get<double>();
                if (!std::isfinite(threshold) || threshold < 0 || threshold > 1)
                    throw InvalidInferenceRequest("INVALID_CONFIDENCE_THRESHOLD", "阈值必须位于 0 到 1");
                inference.confidenceThreshold = static_cast<float>(threshold);
            }
            auto body = serializeResult(service_.infer(inference, modelId, version, metadata["RequestId"].get<std::string>()));
            copyMetadata(metadata, body);
            sendJson(response, 200, body);
        }
        catch (const InvalidInferenceRequest& error) { fail(error.code() == "MODEL_NOT_ACTIVE" ? 409 : 400, error.code(), error.what()); }
        catch (const BackendNotReady& error) { fail(503, "BACKEND_NOT_READY", error.what()); }
        catch (const InferenceOverloaded& error) { fail(503, "SERVICE_BUSY", error.what()); }
        catch (const std::exception& error) { fail(500, "INFERENCE_FAILED", error.what()); }
    });
    server_.set_error_handler([](const httplib::Request&, httplib::Response& response)
    {
        if (response.get_header_value("Content-Type").find("application/json") != std::string::npos) return;
        if (response.status == 404) sendJson(response, 404, makeError("NOT_FOUND", "接口不存在"));
        else if (response.status == 413) sendJson(response, 413, makeError("PAYLOAD_TOO_LARGE", "请求体超过字节上限"));
        else if (response.status == 400) sendJson(response, 400, makeError("INVALID_MULTIPART", "HTTP 或 multipart 请求无效"));
    });
    server_.set_exception_handler([](const httplib::Request&, httplib::Response& response, std::exception_ptr error)
    {
        std::string message = "服务内部错误";
        try { if (error) std::rethrow_exception(error); }
        catch (const std::exception& exception) { message = exception.what(); }
        catch (...) {}
        sendJson(response, 500, makeError("INTERNAL_ERROR", message));
    });
}

} // namespace triton_backend
