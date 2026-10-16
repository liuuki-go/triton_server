#pragma once

#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

struct Detection
{
    // 后处理已经把Engine输出框还原到原图坐标系，HTTP层只负责序列化。
    int classId{};
    float confidence{};
    float x1{};
    float y1{};
    float x2{};
    float y2{};
    std::string className{};
};

enum class PixelFormat
{
    Mono8,
    RGB8Packed,
    BGR8Packed
};

struct RawImage
{
    int width{};
    int height{};
    PixelFormat pixelFormat{PixelFormat::BGR8Packed};
    std::size_t strideBytes{};
    bool bottomUp{};
    // 流水线异步处理期间共享不可变像素数据，避免复制整帧。
    std::shared_ptr<const std::string> data;
};

inline void validateRawImage(const RawImage& image)
{
    if (image.width <= 0 || image.height <= 0)
    {
        throw std::invalid_argument("原始图像宽高必须大于0");
    }

    std::size_t channels = 0;
    switch (image.pixelFormat)
    {
    case PixelFormat::Mono8:
        channels = 1;
        break;
    case PixelFormat::RGB8Packed:
    case PixelFormat::BGR8Packed:
        channels = 3;
        break;
    default:
        throw std::invalid_argument("不支持的原始图像像素格式");
    }

    const auto width = static_cast<std::size_t>(image.width);
    const auto height = static_cast<std::size_t>(image.height);
    const auto maximum = std::numeric_limits<std::size_t>::max();
    if (width > maximum / channels)
    {
        throw std::invalid_argument("原始图像行字节数溢出");
    }
    if (image.strideBytes < width * channels)
    {
        throw std::invalid_argument("原始图像步长小于一行有效像素字节数");
    }
    if (image.strideBytes > maximum / height)
    {
        throw std::invalid_argument("原始图像数据长度溢出");
    }
    if (!image.data || image.data->empty() ||
        image.data->size() != image.strideBytes * height)
    {
        throw std::invalid_argument("原始图像数据长度必须等于strideBytes乘以height");
    }
}

struct PngImage
{
    int width{};
    int height{};
    std::shared_ptr<const std::string> data;
};

inline void validatePngImage(const PngImage& image)
{
    if (image.width <= 0 || image.height <= 0 || !image.data || image.data->size() < 33 ||
        image.data->size() > static_cast<std::size_t>(std::numeric_limits<int>::max()))
    {
        throw std::invalid_argument("PNG 文件为空、尺寸无效或文件头不完整");
    }
    const auto& bytes = *image.data;
    if (bytes.compare(0, 8, "\x89PNG\r\n\x1a\n", 8) != 0 ||
        bytes.compare(8, 8, "\0\0\0\x0dIHDR", 8) != 0)
    {
        throw std::invalid_argument("image 必须包含有效的 PNG 文件头和 IHDR");
    }
    const auto dimension = [&bytes](std::size_t offset)
    {
        std::uint32_t value = 0;
        for (std::size_t index = offset; index < offset + 4; ++index)
            value = (value << 8) | static_cast<unsigned char>(bytes[index]);
        return value;
    };
    if (dimension(16) != static_cast<std::uint32_t>(image.width) ||
        dimension(20) != static_cast<std::uint32_t>(image.height))
    {
        throw std::invalid_argument("Width、Height 必须与 PNG 文件中的尺寸一致");
    }
    if (static_cast<unsigned char>(bytes[24]) > 8)
    {
        throw std::invalid_argument("PNG 仅支持每通道最多8位的像素");
    }
    // 压缩文件的大小不能约束解码内存，在解码前限制 BGR 像素为 64 MiB。
    constexpr std::uint64_t maxDecodedBytes = 64U * 1024U * 1024U;
    if (static_cast<std::uint64_t>(image.width) * image.height > maxDecodedBytes / 3)
    {
        throw std::invalid_argument("PNG 解码后的 BGR 像素超过64 MiB上限");
    }
}

struct InferenceRequest
{
    // Service层接收的业务请求。HTTP JSON等协议细节不会进入推理层。
    std::string imagePath;
    float confidenceThreshold{0.25F};
    std::optional<RawImage> image{};
    std::optional<PngImage> pngImage{};
};

struct InferenceStageTimings
{
    // Service层检查阈值、文件路径或原始像素布局的时间。
    double validationMilliseconds{};
    // 从进入预处理队列到线程取得可用预处理缓冲区的等待时间。
    double preprocessQueueWaitMilliseconds{};
    // 预处理完成后，等待某个GPU槽位取走的时间。
    double gpuQueueWaitMilliseconds{};
    // 为兼容原有性能指标保留，等于上面两个队列等待时间之和。
    double poolWaitMilliseconds{};
    // 解码或原始像素转换、resize、letterbox及融合张量转换的CPU时间。
    double preprocessMilliseconds{};
    // Triton 实现使用 inferenceMilliseconds 记录从提交到响应释放的请求耗时。
    // hostToDeviceMilliseconds/deviceToHostMilliseconds 在当前实现中未单独测量。
    double hostToDeviceMilliseconds{};
    double inferenceMilliseconds{};
    double deviceToHostMilliseconds{};
    // 把模型输出过滤并映射回原图坐标的CPU时间。
    double postprocessMilliseconds{};
    // 从Service开始校验到取得最终检测结果的端到端服务时间。
    double totalMilliseconds{};
};

struct DetectorInferenceResult
{
    // Detector层返回GPU槽位及分阶段指标，供Service补齐总耗时。
    std::size_t poolSlot{};
    InferenceStageTimings timings;
    std::vector<Detection> detections;
};

struct InferenceResult
{
    // Service层的完整结果，HTTP层可直接转换成响应JSON。
    std::string imagePath;
    float confidenceThreshold{};
    double elapsedMilliseconds{};
    std::vector<Detection> detections;
    std::size_t poolSlot{};
    InferenceStageTimings timings;
};

class InferenceOverloaded : public std::runtime_error
{
public:
    // 预处理入口队列已满时抛出，HTTP层将其映射为503 SERVICE_BUSY。
    using std::runtime_error::runtime_error;
};
