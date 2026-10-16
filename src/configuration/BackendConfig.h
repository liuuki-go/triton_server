#pragma once

#include "services/BackendTypes.h"

#include <cstddef>
#include <string>
#include <vector>

namespace triton_backend
{

struct ServerOptions
{
    std::string host{"127.0.0.1"};
    int port{6577};
    std::string schemaVersion{"1.0"};
    std::size_t workerCount{16};
    std::size_t queueCapacity{8};
    std::size_t maxRequestBodyBytes{64ULL * 1024ULL * 1024ULL};
    std::size_t maxMetadataBytes{16384};
    int readTimeoutSeconds{10};
    int writeTimeoutSeconds{30};
    std::vector<std::string> allowedOrigins;
};

struct BackendConfig
{
    ServerOptions server;
    RuntimeConfig runtime;
    std::vector<ModelConfig> models;
    bool autoStart{};
};

BackendConfig loadBackendConfig(const std::string& path);

} // namespace triton_backend
