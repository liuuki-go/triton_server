#include "api/MetadataCodec.h"
#include "domain/InferenceErrors.h"

#include <algorithm>
#include <array>
#include <cctype>
#include <cstdint>
#include <limits>
#include <memory>
#include <regex>
#include <utility>

namespace triton_backend
{

using Json = nlohmann::json;

void sendJson(
    httplib::Response& response,
    int status,
    const Json& body
)
{
    response.status = status;
    response.set_content(
        body.dump(-1, ' ', false, Json::error_handler_t::replace),
        "application/json; charset=utf-8"
    );
}

Json makeError(
    const std::string& code,
    const std::string& message
)
{
    return Json{
        {"ok", false},
        {"error", code},
        {"message", message}
    };
}

static constexpr std::array<const char*, 17> metadataFields{
    "SchemaVersion", "RequestId", "Sequence", "FrameId", "CameraId",
    "ChannelId", "CapturedAtUtc", "Width", "Height", "SourcePixelFormat",
    "ImageEncoding", "ModelId", "RequestedModelVersion", "PixelFormat",
    "StrideBytes", "RowOrder", "ConfidenceThreshold"
};

void copyMetadata(const Json& request, Json& response)
{
    if (!request.is_object())
    {
        return;
    }
    for (const char* field : metadataFields)
    {
        if (request.contains(field))
        {
            response[field] = request[field];
        }
    }
}

static std::string requiredString(const Json& metadata, const char* field)
{
    if (!metadata.contains(field) || !metadata[field].is_string())
    {
        throw InvalidInferenceRequest("INVALID_METADATA", std::string(field) + " 必须是字符串");
    }
    const std::string value = metadata[field].get<std::string>();
    if (value.empty() || value.size() > 256 ||
        std::all_of(value.begin(), value.end(), [](unsigned char character)
        {
            return std::isspace(character) != 0;
        }))
    {
        throw InvalidInferenceRequest("INVALID_METADATA", std::string(field) + " 必须非空且不超过256字节");
    }
    return value;
}

ModelSelection parseModelSelection(const Json& body)
{
    if (!body.is_object() || body.size() != 2 || !body.contains("ModelId") ||
        !body.contains("RequestedModelVersion"))
        throw InvalidInferenceRequest("INVALID_REQUEST", "需要 ModelId 和 RequestedModelVersion 两个字段");
    return ModelSelection{requiredString(body, "ModelId"), requiredString(body, "RequestedModelVersion")};
}

static std::uint64_t requiredUnsigned(const Json& metadata, const char* field)
{
    if (!metadata.contains(field) || !metadata[field].is_number_integer() ||
        (!metadata[field].is_number_unsigned() && metadata[field].get<std::int64_t>() < 0))
    {
        throw InvalidInferenceRequest("INVALID_METADATA", std::string(field) + " 必须是非负整数");
    }
    return metadata[field].get<std::uint64_t>();
}

bool hasMediaType(const std::string& contentType, const std::string& expected, bool requireUtf8)
{
    const auto trim = [](const std::string& value)
    {
        const auto first = value.find_first_not_of(" \t");
        const auto last = value.find_last_not_of(" \t");
        return first == std::string::npos ? std::string{} : value.substr(first, last - first + 1);
    };
    std::string normalized = contentType;
    std::transform(normalized.begin(), normalized.end(), normalized.begin(), [](unsigned char character)
    {
        return static_cast<char>(std::tolower(character));
    });
    const auto firstParameter = normalized.find(';');
    if (trim(normalized.substr(0, firstParameter)) != expected) return false;
    if (!requireUtf8 || firstParameter == std::string::npos) return true;
    for (std::size_t start = firstParameter + 1; start < normalized.size();)
    {
        const auto end = normalized.find(';', start);
        const auto parameter = trim(normalized.substr(start, end - start));
        const auto equals = parameter.find('=');
        if (trim(parameter.substr(0, equals)) == "charset")
        {
            if (equals == std::string::npos) return false;
            auto value = trim(parameter.substr(equals + 1));
            if (value.size() >= 2 && value.front() == '"' && value.back() == '"')
            {
                value = value.substr(1, value.size() - 2);
            }
            if (value != "utf-8" && value != "utf8") return false;
        }
        if (end == std::string::npos) break;
        start = end + 1;
    }
    return true;
}

static bool isUtcTimestamp(const std::string& value)
{
    static const std::regex pattern(
        R"(^([0-9]{4})-([0-9]{2})-([0-9]{2})T([0-9]{2}):([0-9]{2}):([0-9]{2})(\.[0-9]{1,9})?(Z|\+00:00)$)"
    );
    std::smatch match;
    if (!std::regex_match(value, match, pattern)) return false;
    const int year = std::stoi(match[1]);
    const int month = std::stoi(match[2]);
    const int day = std::stoi(match[3]);
    if (year == 0 || month < 1 || month > 12) return false;
    const bool leap = year % 4 == 0 && (year % 100 != 0 || year % 400 == 0);
    const int days[] = {31, 28 + static_cast<int>(leap), 31, 30, 31, 30, 31, 31, 30, 31, 30, 31};
    return day >= 1 && day <= days[month - 1] &&
        std::stoi(match[4]) < 24 && std::stoi(match[5]) < 60 && std::stoi(match[6]) < 60;
}

InferenceRequest parseInferenceImage(const Json& metadata, const std::string& pixels)
{
    for (const char* field : {"SchemaVersion", "RequestId", "CameraId", "ChannelId",
                             "SourcePixelFormat", "ModelId", "RequestedModelVersion"})
    {
        requiredString(metadata, field);
    }
    requiredUnsigned(metadata, "Sequence");
    requiredUnsigned(metadata, "FrameId");
    if (!isUtcTimestamp(requiredString(metadata, "CapturedAtUtc")))
    {
        throw InvalidInferenceRequest("INVALID_METADATA", "CapturedAtUtc 必须是有效的ISO8601 UTC时间");
    }
    const auto encoding = requiredString(metadata, "ImageEncoding");
    if (encoding != "Raw" && encoding != "Png")
    {
        throw InvalidInferenceRequest("UNSUPPORTED_IMAGE_ENCODING", "ImageEncoding 只支持 Png 或 Raw");
    }
    const auto width = requiredUnsigned(metadata, "Width");
    const auto height = requiredUnsigned(metadata, "Height");
    const auto imageError = encoding == "Png" ? "INVALID_PNG_IMAGE" : "INVALID_RAW_IMAGE";
    if (width == 0 || width > std::numeric_limits<int>::max() ||
        height == 0 || height > std::numeric_limits<int>::max())
    {
        throw InvalidInferenceRequest(imageError, "图像尺寸超出范围");
    }
    InferenceRequest request;
    if (encoding == "Png")
    {
        PngImage image{static_cast<int>(width), static_cast<int>(height),
                       std::make_shared<const std::string>(pixels)};
        try { validatePngImage(image); }
        catch (const std::invalid_argument& error)
        {
            throw InvalidInferenceRequest("INVALID_PNG_IMAGE", error.what());
        }
        request.pngImage = std::move(image);
        return request;
    }
    RawImage image;
    const auto stride = requiredUnsigned(metadata, "StrideBytes");
    if (stride > std::numeric_limits<std::size_t>::max())
    {
        throw InvalidInferenceRequest("INVALID_RAW_IMAGE", "原始像素尺寸或行步长超出范围");
    }
    image.width = static_cast<int>(width);
    image.height = static_cast<int>(height);
    image.strideBytes = static_cast<std::size_t>(stride);
    const auto format = requiredString(metadata, "PixelFormat");
    if (format == "Mono8") image.pixelFormat = PixelFormat::Mono8;
    else if (format == "RGB8Packed") image.pixelFormat = PixelFormat::RGB8Packed;
    else if (format == "BGR8Packed") image.pixelFormat = PixelFormat::BGR8Packed;
    else throw InvalidInferenceRequest("UNSUPPORTED_PIXEL_FORMAT", "PixelFormat 只支持 Mono8、RGB8Packed、BGR8Packed");
    const auto rowOrder = requiredString(metadata, "RowOrder");
    if (rowOrder != "TopDown" && rowOrder != "BottomUp")
    {
        throw InvalidInferenceRequest("INVALID_RAW_IMAGE", "RowOrder 只支持 TopDown 或 BottomUp");
    }
    image.bottomUp = rowOrder == "BottomUp";
    // 共享所有权保证HTTP回调等待期间及异步预处理中的像素内存有效。
    image.data = std::make_shared<const std::string>(pixels);
    try
    {
        validateRawImage(image);
    }
    catch (const std::invalid_argument& error)
    {
        throw InvalidInferenceRequest("INVALID_RAW_IMAGE", error.what());
    }
    request.image = std::move(image);
    return request;
}

Json serializeResult(const InferenceResult& result)
{
    Json detections = Json::array();
    for (const auto& detection : result.detections)
    {
        detections.push_back({{"classId", detection.classId}, {"className", detection.className},
            {"confidence", detection.confidence}, {"box", {{"x1", detection.x1},
            {"y1", detection.y1}, {"x2", detection.x2}, {"y2", detection.y2}}}});
    }
    const auto& timings = result.timings;
    return {{"ok", true}, {"appliedConfidenceThreshold", result.confidenceThreshold},
        {"detectionCount", detections.size()}, {"detections", std::move(detections)},
        {"elapsedMs", result.elapsedMilliseconds}, {"timingsMs", {
            {"validation", timings.validationMilliseconds}, {"preprocess", timings.preprocessMilliseconds},
            {"tritonRequest", timings.inferenceMilliseconds}, {"postprocess", timings.postprocessMilliseconds},
            {"total", timings.totalMilliseconds}}}};
}

} // namespace triton_backend
