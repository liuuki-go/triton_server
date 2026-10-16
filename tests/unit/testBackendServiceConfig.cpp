#include "services/impl/BackendService.h"

#include <gtest/gtest.h>

#include <cstdlib>
#include <limits>

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

TEST(BackendServiceConfigurationTest, RejectsInvalidLimitsBeforeCreatingCore)
{
    const ModelConfig model{"best", "1.0.0", "best", 1, 960, {{0, "test"}}};
    EXPECT_THROW(BackendService(RuntimeConfig{"/missing", "/missing", 0, 30}, {model}), std::invalid_argument);
    EXPECT_THROW(BackendService(RuntimeConfig{"/missing", "/missing", 1, 0}, {model}), std::invalid_argument);
    EXPECT_THROW(BackendService(RuntimeConfig{"/missing", "/missing", 1, 30}, {model}), std::invalid_argument);
}

TEST(BackendServiceConfigurationTest, RejectsDuplicateOrInvalidModelsBeforeCreatingCore)
{
    const ModelConfig model{"best", "1.0.0", "best", 1, 960, {{0, "test"}}};
    EXPECT_THROW(BackendService(runtimeConfig(), {}), std::invalid_argument);
    EXPECT_THROW(BackendService(runtimeConfig(), {model, model}), std::invalid_argument);
    auto invalid = model;
    invalid.inputSize = 800;
    EXPECT_THROW(BackendService(runtimeConfig(), {invalid}), std::invalid_argument);
    invalid = model;
    invalid.tritonVersion = -1;
    EXPECT_THROW(BackendService(runtimeConfig(), {invalid}), std::invalid_argument);
    invalid = model;
    invalid.classNames.clear();
    EXPECT_THROW(BackendService(runtimeConfig(), {invalid}), std::invalid_argument);
}

} // namespace
