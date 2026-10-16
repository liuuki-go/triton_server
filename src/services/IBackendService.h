#pragma once

#include "domain/InferenceTypes.h"
#include "domain/InferenceErrors.h"
#include "services/BackendTypes.h"

#include <nlohmann/json.hpp>

#include <string>

namespace triton_backend
{

class IBackendService
{
public:
    virtual ~IBackendService() = default;

    virtual void selectModel(const ModelSelection& selection) = 0;
    virtual InferenceResult infer(
        const InferenceRequest& request,
        const std::string& modelId,
        const std::string& externalVersion,
        const std::string& requestId
    ) = 0;
    virtual nlohmann::json status() const = 0;
};

} // namespace triton_backend
