#include "configuration/BackendConfig.h"

#include <nlohmann/json.hpp>
#include <yaml-cpp/yaml.h>

#include <algorithm>
#include <charconv>
#include <filesystem>
#include <fstream>
#include <set>
#include <stdexcept>

namespace triton_backend
{
namespace
{

std::size_t integer(const YAML::Node& node, const char* key, std::size_t fallback,
                    std::size_t minimum, std::size_t maximum)
{
    if (!node[key]) return fallback;
    const auto value = node[key];
    const std::string text = value.IsScalar() ? value.Scalar() : "";
    std::size_t parsed{};
    const auto result = std::from_chars(text.data(), text.data() + text.size(), parsed);
    if ((value.Tag() != "?" && value.Tag() != "tag:yaml.org,2002:int") ||
        text.empty() || result.ec != std::errc{} || result.ptr != text.data() + text.size() ||
        !std::all_of(text.begin(), text.end(), [](char c) { return c >= '0' && c <= '9'; }) ||
        parsed < minimum || parsed > maximum)
    {
        throw std::invalid_argument(std::string("配置整数超出范围：") + key);
    }
    return parsed;
}

std::uint64_t modelInteger(const nlohmann::json& value, const char* field,
                           std::uint64_t minimum, std::uint64_t maximum)
{
    if (!value.is_number_integer() ||
        (!value.is_number_unsigned() && value.get<std::int64_t>() < 0))
        throw std::invalid_argument(std::string("模型配置必须为非负整数：") + field);
    const auto parsed = value.get<std::uint64_t>();
    if (parsed < minimum || parsed > maximum)
        throw std::invalid_argument(std::string("模型配置整数超出范围：") + field);
    return parsed;
}

std::string text(const YAML::Node& node, const char* key, const std::string& fallback = {})
{
    if (!node[key]) return fallback;
    const auto scalar = node[key];
    bool boolean{};
    double numeric{};
    if (!scalar.IsScalar() || (scalar.Tag() != "!" && scalar.Tag() != "tag:yaml.org,2002:str" &&
        (YAML::convert<bool>::decode(scalar, boolean) || YAML::convert<double>::decode(scalar, numeric))))
        throw std::invalid_argument(std::string("配置必须为字符串：") + key);
    const auto value = node[key].as<std::string>();
    if (value.empty()) throw std::invalid_argument(std::string("配置不能为空：") + key);
    return value;
}

std::string relativePath(const std::filesystem::path& directory, const std::string& value)
{
    auto path = std::filesystem::path(value);
    if (path.is_relative()) path = directory / path;
    return std::filesystem::absolute(path).lexically_normal().string();
}

} // namespace

BackendConfig loadBackendConfig(const std::string& path)
{
    const auto document = YAML::LoadFile(path);
    if (!document.IsMap() || !document["server"].IsMap() || !document["triton"].IsMap())
    {
        throw std::invalid_argument("配置需要 server 和 triton 对象");
    }
    const auto directory = std::filesystem::absolute(path).parent_path();
    const auto server = document["server"];
    const auto triton = document["triton"];
    BackendConfig config;
    config.server.host = text(server, "host", "127.0.0.1");
    if (config.server.host != "127.0.0.1" && config.server.host != "localhost" &&
        config.server.host != "::1" && config.server.host != "0.0.0.0")
        throw std::invalid_argument("监听地址仅支持 127.0.0.1、localhost、::1 或 0.0.0.0");
    config.server.port = static_cast<int>(integer(server, "port", 6577, 1, 65535));
    config.server.schemaVersion = text(server, "schemaVersion", "1.0");
    if (config.server.schemaVersion != "1.0") throw std::invalid_argument("只支持协议 1.0");
    config.server.workerCount = integer(server, "workerCount", 16, 2, 1024);
    config.server.queueCapacity = integer(server, "queueCapacity", 8, 1, 65536);
    config.server.maxRequestBodyBytes = integer(server, "maxRequestBodyBytes", 67108864, 1, 67108864);
    config.server.maxMetadataBytes = integer(server, "maxMetadataBytes", 16384, 1, 65536);
    config.server.readTimeoutSeconds = static_cast<int>(integer(server, "readTimeoutSeconds", 10, 1, 3600));
    config.server.writeTimeoutSeconds = static_cast<int>(integer(server, "writeTimeoutSeconds", 30, 1, 3600));
    if (server["allowedOrigins"])
    {
        if (!server["allowedOrigins"].IsSequence()) throw std::invalid_argument("allowedOrigins 必须为数组");
        for (const auto& origin : server["allowedOrigins"]) config.server.allowedOrigins.push_back(origin.as<std::string>());
    }
    config.runtime.root = relativePath(directory, text(triton, "root", "/opt/triton/2.71.0"));
    config.runtime.repository = relativePath(directory, text(triton, "repository", "../models"));
    config.runtime.maxInFlight = integer(triton, "maxInFlight", 8, 1, 1023);
    if (config.runtime.maxInFlight >= config.server.workerCount)
        throw std::invalid_argument("workerCount 必须大于 maxInFlight，为控制请求保留线程");
    config.runtime.stopTimeoutSeconds = static_cast<int>(integer(triton, "stopTimeoutSeconds", 30, 1, 3600));
    config.autoStart = document["autoStart"] ? document["autoStart"].as<bool>() : false;

    std::ifstream source(relativePath(directory, text(triton, "modelConfigPath", "models.json")));
    if (!source) throw std::invalid_argument("无法读取模型配置");
    const auto modelDocument = nlohmann::json::parse(source);
    const auto& models = modelDocument.at("models");
    if (!models.is_array() || models.empty()) throw std::invalid_argument("至少需要一个模型配置");
    for (const auto& value : models)
    {
        ModelConfig model;
        model.id = value.at("id").get<std::string>();
        model.externalVersion = value.at("businessVersion").get<std::string>();
        model.tritonName = value.at("tritonName").get<std::string>();
        const auto version = value.at("tritonVersion");
        const auto size = value.at("inputSize");
        if (!version.is_number_integer() || !size.is_number_integer() ||
            (version.is_number_unsigned() && version.get<std::uint64_t>() > static_cast<std::uint64_t>(INT64_MAX)) ||
            (size != 640 && size != 960))
            throw std::invalid_argument("Triton 版本和输入尺寸必须为有效整数");
        model.tritonVersion = value.at("tritonVersion").get<std::int64_t>();
        model.inputSize = value.at("inputSize").get<int>();
        if (model.id.empty() || model.id.size() > 256 || model.externalVersion.empty() ||
            model.externalVersion.size() > 256 || model.tritonName.empty() || model.tritonVersion < 1 ||
            (model.inputSize != 640 && model.inputSize != 960))
            throw std::invalid_argument("模型标识、版本、尺寸或重复配置无效");
        if (value.contains("instanceCount"))
            model.instanceCount = modelInteger(value.at("instanceCount"), "instanceCount", 1, 32);
        if (value.contains("batching"))
        {
            const auto& batching = value.at("batching");
            if (!batching.is_object()) throw std::invalid_argument("batching 必须为对象");
            if (batching.contains("maxBatchSize"))
                model.batching.maxBatchSize = modelInteger(batching.at("maxBatchSize"), "maxBatchSize", 0, 1024);
            if (batching.contains("maxQueueDelayMicroseconds"))
                model.batching.maxQueueDelayMicroseconds = modelInteger(
                    batching.at("maxQueueDelayMicroseconds"), "maxQueueDelayMicroseconds", 0, 1000000);
            if (batching.contains("preferredBatchSizes"))
            {
                const auto& preferred = batching.at("preferredBatchSizes");
                if (!preferred.is_array()) throw std::invalid_argument("preferredBatchSizes 必须为数组");
                for (const auto& size : preferred)
                    model.batching.preferredBatchSizes.push_back(modelInteger(size, "preferredBatchSizes", 1, 1024));
            }
        }
        const auto& classes = value.at("classNames");
        if (!classes.is_object() || classes.empty()) throw std::invalid_argument("classNames 必须是非空对象");
        for (const auto& entry : classes.items())
        {
            int id{};
            const auto key = entry.key();
            const auto parsed = std::from_chars(key.data(), key.data() + key.size(), id);
            const auto name = entry.value().get<std::string>();
            if (parsed.ec != std::errc{} || parsed.ptr != key.data() + key.size() || id < 0 || name.empty() ||
                !model.classNames.emplace(id, name).second)
                throw std::invalid_argument("类别 ID 或名称无效");
        }
        config.models.push_back(std::move(model));
    }
    validateModelCatalog(config.models);
    return config;
}

} // namespace triton_backend
