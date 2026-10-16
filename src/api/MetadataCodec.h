#pragma once

#include "domain/InferenceTypes.h"
#include "services/BackendTypes.h"

#include <httplib.h>
#include <nlohmann/json.hpp>

#include <string>

namespace triton_backend
{

void sendJson(httplib::Response& response, int status, const nlohmann::json& body);
nlohmann::json makeError(const std::string& code, const std::string& message);
void copyMetadata(const nlohmann::json& request, nlohmann::json& response);
bool hasMediaType(const std::string& contentType, const std::string& expected, bool requireUtf8 = false);
InferenceRequest parseInferenceImage(const nlohmann::json& metadata, const std::string& bytes);
ModelSelection parseModelSelection(const nlohmann::json& body);
nlohmann::json serializeResult(const InferenceResult& result);

} // namespace triton_backend
