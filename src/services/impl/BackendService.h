#pragma once

#include "services/BackendTypes.h"
#include "services/IBackendService.h"

#include <memory>
#include <vector>

namespace triton_backend
{

class BackendService final : public IBackendService
{
public:
    BackendService(RuntimeConfig runtime, std::vector<ModelConfig> models);
    ~BackendService() override;

    BackendService(const BackendService&) = delete;
    BackendService& operator=(const BackendService&) = delete;

    void selectModel(const ModelSelection& selection) override;
    void stop();
    bool isReady() const;
    InferenceResult infer(
        const InferenceRequest& request,
        const std::string& modelId,
        const std::string& externalVersion,
        const std::string& requestId
    ) override;
    nlohmann::json status() const override;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace triton_backend
