#include "configuration/BackendConfig.h"

#include <gtest/gtest.h>
#include <nlohmann/json.hpp>

#include <filesystem>
#include <fstream>
#include <limits>
#include <unistd.h>

namespace
{

class BackendConfigTest : public ::testing::Test
{
protected:
    void SetUp() override
    {
        directory = std::filesystem::temp_directory_path() /
            ("triton-config-test-" + std::to_string(getpid()));
        std::filesystem::create_directories(directory);
        model = {{"id", "best"}, {"businessVersion", "1.0.0"}, {"tritonName", "best"},
            {"tritonVersion", 1}, {"inputSize", 960}, {"classNames", {{"0", "airflow_print"}}}};
        writeModels(nlohmann::json::array({model}));
    }
    void TearDown() override { std::filesystem::remove_all(directory); }
    void writeModels(const nlohmann::json& models)
    {
        std::ofstream(directory / "models.json") << nlohmann::json{{"models", models}}.dump();
    }
    std::string writeConfig(const std::string& server = "host: '127.0.0.1'\n  port: 6577",
                            const std::string& runtime = "maxInFlight: 8",
                            const std::string& modelConfig = "'models.json'")
    {
        const auto path = directory / "server.yaml";
        std::ofstream(path) << "server:\n  " << server << "\ntriton:\n  root: '/opt/triton/2.71.0'\n"
            << "  repository: '../models'\n  modelConfigPath: " << modelConfig << "\n  " << runtime << '\n';
        return path.string();
    }
    std::filesystem::path directory;
    nlohmann::json model;
};

TEST_F(BackendConfigTest, ResolvesRepositoryAndKeepsBusinessVersionSeparate)
{
    const auto config = triton_backend::loadBackendConfig(writeConfig());
    EXPECT_EQ(config.runtime.repository, (directory.parent_path() / "models").string());
    EXPECT_EQ(config.models.front().externalVersion, "1.0.0");
    EXPECT_EQ(config.models.front().tritonVersion, 1);
    EXPECT_EQ(config.models.front().classNames.at(0), "airflow_print");
    EXPECT_EQ(config.models.front().batching.maxBatchSize, 0U);
    EXPECT_TRUE(config.models.front().batching.preferredBatchSizes.empty());
    EXPECT_EQ(config.models.front().batching.maxQueueDelayMicroseconds, 0U);
    EXPECT_EQ(config.models.front().instanceCount, 1U);
    EXPECT_FALSE(config.autoStart);
}

TEST_F(BackendConfigTest, LoadsExplicitBatchingAndInstanceConfiguration)
{
    model["batching"] = {{"maxBatchSize", 8}, {"preferredBatchSizes", {4}}, {"maxQueueDelayMicroseconds", 10000}};
    model["instanceCount"] = 2;
    writeModels(nlohmann::json::array({model}));
    const auto config = triton_backend::loadBackendConfig(writeConfig());
    EXPECT_EQ(config.models.front().batching.maxBatchSize, 8U);
    EXPECT_EQ(config.models.front().batching.preferredBatchSizes, std::vector<std::size_t>{4});
    EXPECT_EQ(config.models.front().batching.maxQueueDelayMicroseconds, 10000U);
    EXPECT_EQ(config.models.front().instanceCount, 2U);

    model["batching"] = nlohmann::json::object();
    model.erase("instanceCount");
    writeModels(nlohmann::json::array({model}));
    const auto legacy = triton_backend::loadBackendConfig(writeConfig());
    EXPECT_EQ(legacy.models.front().batching.maxBatchSize, 0U);
    EXPECT_TRUE(legacy.models.front().batching.preferredBatchSizes.empty());
    EXPECT_EQ(legacy.models.front().batching.maxQueueDelayMicroseconds, 0U);
    EXPECT_EQ(legacy.models.front().instanceCount, 1U);

    model["batching"] = {{"maxBatchSize", 8}};
    writeModels(nlohmann::json::array({model}));
    const auto unspecified = triton_backend::loadBackendConfig(writeConfig());
    EXPECT_TRUE(unspecified.models.front().batching.preferredBatchSizes.empty());
    EXPECT_EQ(unspecified.models.front().batching.maxQueueDelayMicroseconds, 0U);

    model["batching"] = {{"maxBatchSize", 1024}, {"preferredBatchSizes", {1, 4, 1024}},
                         {"maxQueueDelayMicroseconds", 1000000}};
    model["instanceCount"] = 32;
    writeModels(nlohmann::json::array({model}));
    EXPECT_NO_THROW(triton_backend::loadBackendConfig(writeConfig()));
}

TEST_F(BackendConfigTest, RejectsInvalidBatchingIntegerTypesAndRanges)
{
    const std::vector<nlohmann::json> invalidValues{
        true, false, -1, 1.5, "4", nullptr, nlohmann::json::array(),
        std::numeric_limits<std::uint64_t>::max()};
    for (const char* field : {"maxBatchSize", "maxQueueDelayMicroseconds", "instanceCount"})
    {
        for (const auto& invalid : invalidValues)
        {
            SCOPED_TRACE(std::string(field) + ": " + invalid.dump());
            auto candidate = model;
            if (std::string(field) == "instanceCount") candidate[field] = invalid;
            else candidate["batching"] = {{field, invalid}};
            writeModels(nlohmann::json::array({candidate}));
            EXPECT_THROW(triton_backend::loadBackendConfig(writeConfig()), std::exception);
        }
    }
    for (const auto& invalid : std::vector<nlohmann::json>{
             {{"batching", {{"maxBatchSize", 1025}}}},
             {{"batching", {{"maxBatchSize", 8}, {"maxQueueDelayMicroseconds", 1000001}}}},
             {{"instanceCount", 0}}, {{"instanceCount", 33}}})
    {
        auto candidate = model;
        candidate.update(invalid);
        writeModels(nlohmann::json::array({candidate}));
        EXPECT_THROW(triton_backend::loadBackendConfig(writeConfig()), std::exception);
    }
}

TEST_F(BackendConfigTest, RejectsInvalidPreferredBatchSizesAndDisabledBatchingOptions)
{
    for (const auto& invalid : std::vector<nlohmann::json>{
             nullptr, true, 4, "4", nlohmann::json::object(),
             {0}, {-1}, {true}, {1.5}, {"4"}, {4, 4}, {9},
             {std::numeric_limits<std::uint64_t>::max()}})
    {
        SCOPED_TRACE(invalid.dump());
        auto candidate = model;
        candidate["batching"] = {{"maxBatchSize", 8}, {"preferredBatchSizes", invalid}};
        writeModels(nlohmann::json::array({candidate}));
        EXPECT_THROW(triton_backend::loadBackendConfig(writeConfig()), std::exception);
    }
    for (const auto& invalid : std::vector<nlohmann::json>{
             nullptr, true, nlohmann::json::array(),
             {{"maxBatchSize", 0}, {"preferredBatchSizes", {1}}},
             {{"maxBatchSize", 0}, {"maxQueueDelayMicroseconds", 1}}})
    {
        auto candidate = model;
        candidate["batching"] = invalid;
        writeModels(nlohmann::json::array({candidate}));
        EXPECT_THROW(triton_backend::loadBackendConfig(writeConfig()), std::exception);
    }
}

TEST_F(BackendConfigTest, ValidatesBatchingForDirectModelConstruction)
{
    const triton_backend::ModelConfig legacy{"best", "1.0.0", "best", 1, 960, {{0, "test"}}};
    EXPECT_NO_THROW(triton_backend::validateModelCatalog({legacy}));
    auto candidate = legacy;
    candidate.batching = {8, {4}, 10000};
    candidate.instanceCount = 2;
    EXPECT_NO_THROW(triton_backend::validateModelCatalog({candidate}));
    for (const auto& invalid : std::vector<triton_backend::ModelBatchingConfig>{
             {1025, {}, 0}, {0, {1}, 0}, {0, {}, 1}, {8, {0}, 0},
             {8, {4, 4}, 0}, {8, {9}, 0}, {8, {}, 1000001}})
    {
        candidate.batching = invalid;
        EXPECT_THROW(triton_backend::validateModelCatalog({candidate}), std::invalid_argument);
    }
    candidate.batching = {};
    for (const auto count : {0U, 33U})
    {
        candidate.instanceCount = count;
        EXPECT_THROW(triton_backend::validateModelCatalog({candidate}), std::invalid_argument);
    }
}

TEST_F(BackendConfigTest, RejectsInvalidAndQuotedIntegers)
{
    for (const auto& value : {"-1", "0", "65536", "1.5", "'6577'", "true"})
        EXPECT_THROW(triton_backend::loadBackendConfig(writeConfig(std::string("port: ") + value)), std::exception);
}

TEST_F(BackendConfigTest, RequiresControlThreadCapacity)
{
    EXPECT_THROW(triton_backend::loadBackendConfig(writeConfig("workerCount: 8", "maxInFlight: 8")), std::exception);
}

TEST_F(BackendConfigTest, AcceptsWildcardAndLoopbackBindings)
{
    for (const char* host : {"0.0.0.0", "127.0.0.1", "localhost", "::1"})
    {
        SCOPED_TRACE(host);
        const auto config = triton_backend::loadBackendConfig(writeConfig(std::string("host: '") + host + "'"));
        EXPECT_EQ(config.server.host, host);
    }
}

TEST_F(BackendConfigTest, DefaultsMissingHostToLoopback)
{
    const auto config = triton_backend::loadBackendConfig(writeConfig("port: 6577"));
    EXPECT_EQ(config.server.host, "127.0.0.1");
}

TEST_F(BackendConfigTest, RejectsUnknownProtocol)
{
    EXPECT_THROW(triton_backend::loadBackendConfig(writeConfig("schemaVersion: '2.0'")), std::exception);
}

TEST_F(BackendConfigTest, RejectsUnsupportedBindings)
{
    for (const char* host : {"192.168.1.10", "::", "backend.example.com", "0.0.0.256"})
    {
        SCOPED_TRACE(host);
        EXPECT_THROW(triton_backend::loadBackendConfig(writeConfig(std::string("host: '") + host + "'")),
            std::invalid_argument);
    }
}

TEST_F(BackendConfigTest, RejectsDuplicateModelsAndBadClassMapping)
{
    writeModels(nlohmann::json::array({model, model}));
    EXPECT_THROW(triton_backend::loadBackendConfig(writeConfig()), std::exception);
    model["classNames"] = {{"-1", "invalid"}};
    writeModels(nlohmann::json::array({model}));
    EXPECT_THROW(triton_backend::loadBackendConfig(writeConfig()), std::exception);
    model["classNames"] = {{"0", ""}};
    writeModels(nlohmann::json::array({model}));
    EXPECT_THROW(triton_backend::loadBackendConfig(writeConfig()), std::exception);
}

TEST_F(BackendConfigTest, AcceptsMultipleVersionsAndRejectsAmbiguousOrUnsafeMappings)
{
    auto next = model;
    next["businessVersion"] = "2.0.0";
    next["tritonVersion"] = 2;
    writeModels(nlohmann::json::array({model, next}));
    EXPECT_EQ(triton_backend::loadBackendConfig(writeConfig()).models.size(), 2U);
    next["tritonVersion"] = 1;
    writeModels(nlohmann::json::array({model, next}));
    EXPECT_THROW(triton_backend::loadBackendConfig(writeConfig()), std::exception);
    next["tritonVersion"] = 2;
    next["tritonName"] = "different-name";
    writeModels(nlohmann::json::array({model, next}));
    EXPECT_THROW(triton_backend::loadBackendConfig(writeConfig()), std::exception);
    next["tritonName"] = "../best";
    writeModels(nlohmann::json::array({next}));
    EXPECT_THROW(triton_backend::loadBackendConfig(writeConfig()), std::exception);
}

TEST_F(BackendConfigTest, RejectsInvalidModelShapeAndMissingConfig)
{
    model["inputSize"] = 800;
    writeModels(nlohmann::json::array({model}));
    EXPECT_THROW(triton_backend::loadBackendConfig(writeConfig()), std::exception);
    std::filesystem::remove(directory / "models.json");
    EXPECT_THROW(triton_backend::loadBackendConfig(writeConfig()), std::exception);
}

TEST_F(BackendConfigTest, RejectsFractionalAndOverflowedModelNumbers)
{
    model["tritonVersion"] = 1.5;
    writeModels(nlohmann::json::array({model}));
    EXPECT_THROW(triton_backend::loadBackendConfig(writeConfig()), std::exception);
    model["tritonVersion"] = std::numeric_limits<std::uint64_t>::max();
    writeModels(nlohmann::json::array({model}));
    EXPECT_THROW(triton_backend::loadBackendConfig(writeConfig()), std::exception);
    model["tritonVersion"] = 1;
    model["inputSize"] = 960.5;
    writeModels(nlohmann::json::array({model}));
    EXPECT_THROW(triton_backend::loadBackendConfig(writeConfig()), std::exception);
}

TEST_F(BackendConfigTest, RejectsNumericSchemaAndBooleanPath)
{
    EXPECT_THROW(triton_backend::loadBackendConfig(writeConfig("schemaVersion: 1.0")), std::exception);
    EXPECT_THROW(triton_backend::loadBackendConfig(writeConfig("port: 6577", "maxInFlight: 8", "true")), std::exception);
}

} // namespace
