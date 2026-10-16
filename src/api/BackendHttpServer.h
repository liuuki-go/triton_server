#pragma once

#include "configuration/BackendConfig.h"
#include "services/IBackendService.h"

#include <httplib.h>

#include <chrono>

namespace triton_backend
{

class BackendHttpServer
{
public:
    BackendHttpServer(IBackendService& service, const BackendConfig& config);
    int bind();
    bool listenAfterBind();
    void stop();

private:
    void registerRoutes();
    nlohmann::json status(bool simple = false) const;
    nlohmann::json selectionStatus() const;
    IBackendService& service_;
    BackendConfig config_;
    httplib::Server server_;
    std::chrono::steady_clock::time_point startedAt_{std::chrono::steady_clock::now()};
};

} // namespace triton_backend
