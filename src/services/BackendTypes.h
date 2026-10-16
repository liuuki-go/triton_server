#pragma once

#include <cstddef>
#include <cstdint>
#include <algorithm>
#include <cctype>
#include <map>
#include <set>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace triton_backend
{

struct ModelBatchingConfig
{
    std::size_t maxBatchSize{0};
    std::vector<std::size_t> preferredBatchSizes;
    std::uint64_t maxQueueDelayMicroseconds{0};
};

struct ModelConfig
{
    std::string id;
    std::string externalVersion;
    std::string tritonName;
    std::int64_t tritonVersion{1};
    int inputSize{640};
    std::map<int, std::string> classNames;
    ModelBatchingConfig batching{};
    std::size_t instanceCount{1};
};

struct RuntimeConfig
{
    std::string root;
    std::string repository;
    std::size_t maxInFlight{8};
    int stopTimeoutSeconds{30};
};

struct ModelSelection
{
    std::string modelId;
    std::string version;
};

inline void validateModelCatalog(const std::vector<ModelConfig>& models)
{
    if (models.empty()) throw std::invalid_argument("至少需要一个模型版本配置");
    std::set<std::pair<std::string, std::string>> versions;
    std::set<std::pair<std::string, std::int64_t>> engines;
    std::map<std::string, std::string> names;
    std::map<std::string, std::string> ids;
    for (const auto& model : models)
    {
        const bool validName = !model.tritonName.empty() && model.tritonName.front() != '.' &&
            std::all_of(model.tritonName.begin(), model.tritonName.end(), [](unsigned char character)
            { return std::isalnum(character) || character == '_' || character == '-' || character == '.'; });
        if (model.id.empty() || model.id.size() > 256 || model.externalVersion.empty() ||
            model.externalVersion.size() > 256 || !validName || model.tritonVersion <= 0 ||
            (model.inputSize != 640 && model.inputSize != 960) || model.classNames.empty() ||
            !versions.emplace(model.id, model.externalVersion).second ||
            !engines.emplace(model.tritonName, model.tritonVersion).second)
            throw std::invalid_argument("模型身份、版本、输入尺寸、类别配置或重复版本无效");
        const auto name = names.emplace(model.id, model.tritonName);
        const auto id = ids.emplace(model.tritonName, model.id);
        if (name.first->second != model.tritonName || id.first->second != model.id)
            throw std::invalid_argument("同一业务模型与 Triton 模型名称必须一一对应");
        for (const auto& item : model.classNames)
            if (item.first < 0 || item.second.empty()) throw std::invalid_argument("模型类别配置无效");
        if (model.batching.maxBatchSize > 1024 || model.instanceCount == 0 || model.instanceCount > 32 ||
            model.batching.maxQueueDelayMicroseconds > 1000000 ||
            (model.batching.maxBatchSize == 0 &&
             (!model.batching.preferredBatchSizes.empty() || model.batching.maxQueueDelayMicroseconds != 0)))
            throw std::invalid_argument("模型批量上限、等待时间或实例数量无效");
        std::set<std::size_t> preferredBatchSizes;
        for (const auto size : model.batching.preferredBatchSizes)
            if (size == 0 || size > model.batching.maxBatchSize || !preferredBatchSizes.insert(size).second)
                throw std::invalid_argument("首选批量必须为正数、互不重复且不超过批量上限");
    }
}

} // namespace triton_backend
