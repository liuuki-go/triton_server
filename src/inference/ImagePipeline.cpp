#include "inference/ImagePipeline.h"
#include "inference/RawImagePreprocessing.h"
#include "domain/InferenceErrors.h"

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace triton_backend
{

static PreparedImage prepareBgrImage(const cv::Mat& source, int inputSize)
{
    if (inputSize != 640 && inputSize != 960)
    {
        throw std::invalid_argument("模型输入尺寸只支持 640 或 960");
    }
    const float scale = std::min(
        static_cast<float>(inputSize) / source.cols,
        static_cast<float>(inputSize) / source.rows
    );
    const int resizedWidth = std::max(1, static_cast<int>(std::round(source.cols * scale)));
    const int resizedHeight = std::max(1, static_cast<int>(std::round(source.rows * scale)));
    cv::Mat resized;
    cv::resize(source, resized, cv::Size(resizedWidth, resizedHeight), 0, 0, cv::INTER_LINEAR);

    const int padLeft = (inputSize - resizedWidth) / 2;
    const int padTop = (inputSize - resizedHeight) / 2;
    const std::size_t channelSize = static_cast<std::size_t>(inputSize) * inputSize;
    constexpr float inputScale = 1.0F / 255.0F;
    PreparedImage prepared{
        std::vector<float>(3 * channelSize, 114.0F * inputScale),
        LetterboxInfo{scale, padLeft, padTop, source.cols, source.rows}
    };
    float* red = prepared.tensor.data();
    float* green = red + channelSize;
    float* blue = green + channelSize;
    for (int row = 0; row < resizedHeight; ++row)
    {
        const cv::Vec3b* pixels = resized.ptr<cv::Vec3b>(row);
        const std::size_t outputRow = static_cast<std::size_t>(row + padTop) * inputSize + padLeft;
        for (int column = 0; column < resizedWidth; ++column)
        {
            const std::size_t index = outputRow + column;
            red[index] = pixels[column][2] * inputScale;
            green[index] = pixels[column][1] * inputScale;
            blue[index] = pixels[column][0] * inputScale;
        }
    }
    return prepared;
}

PreparedImage prepareImage(const RawImage& image, int inputSize)
{
    return prepareBgrImage(rawImageToBgr(image), inputSize);
}

PreparedImage prepareImage(const PngImage& image, int inputSize)
{
    cv::Mat decoded;
    try
    {
        validatePngImage(image);
        const cv::Mat encoded(1, static_cast<int>(image.data->size()), CV_8UC1,
                              const_cast<char*>(image.data->data()));
        // 忽略透明度和 EXIF 方向，坐标始终对应上传 PNG 的像素宽高。
        decoded = cv::imdecode(encoded, cv::IMREAD_COLOR | cv::IMREAD_IGNORE_ORIENTATION);
    }
    catch (const cv::Exception& error)
    {
        throw InvalidInferenceRequest("INVALID_PNG_IMAGE", error.what());
    }
    catch (const std::invalid_argument& error)
    {
        throw InvalidInferenceRequest("INVALID_PNG_IMAGE", error.what());
    }
    if (decoded.empty() || decoded.cols != image.width || decoded.rows != image.height)
    {
        throw InvalidInferenceRequest("INVALID_PNG_IMAGE", "PNG 解码失败或解码尺寸与 metadata 不一致");
    }
    return prepareBgrImage(decoded, inputSize);
}

std::vector<Detection> decodeDetections(
    const std::vector<float>& output,
    const LetterboxInfo& info,
    float confidenceThreshold
)
{
    if (output.size() != 300U * 6U)
    {
        throw std::runtime_error("模型输出必须为 FP32 [1,300,6]");
    }
    if (!std::isfinite(confidenceThreshold) || confidenceThreshold < 0 || confidenceThreshold > 1 ||
        !std::isfinite(info.scale) || info.scale <= 0 || info.originalWidth <= 0 || info.originalHeight <= 0)
    {
        throw std::invalid_argument("后处理阈值或图像变换参数无效");
    }
    std::vector<Detection> detections;
    detections.reserve(300);
    for (std::size_t index = 0; index < 300; ++index)
    {
        const float* row = output.data() + index * 6;
        if (!std::isfinite(row[4]) || row[4] < confidenceThreshold)
        {
            continue;
        }
        if (!std::all_of(row, row + 6, [](float value) { return std::isfinite(value); }) ||
            row[4] > 1.0F || row[5] < 0 ||
            static_cast<double>(row[5]) > std::numeric_limits<int>::max() || std::round(row[5]) != row[5])
        {
            throw std::runtime_error("模型输出了非法检测数值或类别 ID");
        }
        const float x1 = std::clamp((row[0] - info.padLeft) / info.scale, 0.0F,
                                    static_cast<float>(info.originalWidth - 1));
        const float y1 = std::clamp((row[1] - info.padTop) / info.scale, 0.0F,
                                    static_cast<float>(info.originalHeight - 1));
        const float x2 = std::clamp((row[2] - info.padLeft) / info.scale, 0.0F,
                                    static_cast<float>(info.originalWidth - 1));
        const float y2 = std::clamp((row[3] - info.padTop) / info.scale, 0.0F,
                                    static_cast<float>(info.originalHeight - 1));
        if (x2 <= x1 || y2 <= y1)
        {
            continue;
        }
        detections.push_back(Detection{static_cast<int>(row[5]), row[4], x1, y1, x2, y2, {}});
    }
    return detections;
}

} // namespace triton_backend
