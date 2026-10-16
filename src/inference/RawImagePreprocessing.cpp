#include "inference/RawImagePreprocessing.h"

#include <opencv2/imgproc.hpp>

cv::Mat rawImageToBgr(const RawImage& image)
{
    validateRawImage(image);
    const int type = image.pixelFormat == PixelFormat::Mono8
        ? CV_8UC1 : CV_8UC3;
    // OpenCV需要非const地址，转换操作只读取这段共享像素。
    const cv::Mat pixels(
        image.height,
        image.width,
        type,
        const_cast<char*>(image.data->data()),
        image.strideBytes
    );
    cv::Mat bgr;
    if (image.pixelFormat == PixelFormat::Mono8)
    {
        cv::cvtColor(pixels, bgr, cv::COLOR_GRAY2BGR);
    }
    else if (image.pixelFormat == PixelFormat::RGB8Packed)
    {
        cv::cvtColor(pixels, bgr, cv::COLOR_RGB2BGR);
    }
    else
    {
        bgr = pixels;
    }

    if (image.bottomUp)
    {
        cv::Mat topDown;
        cv::flip(bgr, topDown, 0);
        return topDown;
    }
    return bgr;
}
