#include "inference/ImagePipeline.h"
#include "domain/InferenceErrors.h"

#include <gtest/gtest.h>
#include <opencv2/imgcodecs.hpp>

#include <algorithm>
#include <limits>
#include <memory>
#include <string>

using namespace triton_backend;

namespace
{

RawImage solidImage(int width, int height, PixelFormat format, bool bottomUp, std::size_t padding)
{
    const std::size_t channels = format == PixelFormat::Mono8 ? 1 : 3;
    const std::size_t stride = width * channels + padding;
    std::string bytes(stride * height, '\xA5');
    for (int row = 0; row < height; ++row)
    {
        for (int column = 0; column < width; ++column)
        {
            const std::size_t offset = row * stride + column * channels;
            bytes[offset] = format == PixelFormat::RGB8Packed ? 30 : 10;
            if (channels == 3)
            {
                bytes[offset + 1] = 20;
                bytes[offset + 2] = format == PixelFormat::RGB8Packed ? 10 : 30;
            }
        }
    }
    return RawImage{width, height, format, stride, bottomUp,
                    std::make_shared<const std::string>(std::move(bytes))};
}

TEST(ImagePipelineTest, RawFormatsBecomeNormalizedRgbPlanesWithLetterbox)
{
    for (const auto format : {PixelFormat::Mono8, PixelFormat::RGB8Packed, PixelFormat::BGR8Packed})
    {
        for (bool bottomUp : {false, true})
        {
            const RawImage image = solidImage(4, 2, format, bottomUp, 3);
            const auto originalBytes = *image.data;
            const PreparedImage prepared = prepareImage(image, 640);
            ASSERT_EQ(prepared.tensor.size(), 3U * 640U * 640U);
            EXPECT_FLOAT_EQ(prepared.letterbox.scale, 160.0F);
            EXPECT_EQ(prepared.letterbox.padLeft, 0);
            EXPECT_EQ(prepared.letterbox.padTop, 160);
            const std::size_t plane = 640U * 640U;
            const std::size_t first = 160U * 640U;
            EXPECT_FLOAT_EQ(prepared.tensor[0], 114.0F / 255.0F);
            EXPECT_FLOAT_EQ(prepared.tensor[first], (format == PixelFormat::Mono8 ? 10.0F : 30.0F) / 255.0F);
            EXPECT_FLOAT_EQ(prepared.tensor[first + plane], (format == PixelFormat::Mono8 ? 10.0F : 20.0F) / 255.0F);
            EXPECT_FLOAT_EQ(prepared.tensor[first + 2 * plane], 10.0F / 255.0F);
            EXPECT_EQ(*image.data, originalBytes);
        }
    }
}

TEST(ImagePipelineTest, PngGrayColorAndAlphaMatchRawPixelsForBothInputSizes)
{
    for (const int type : {CV_8UC1, CV_8UC3, CV_8UC4})
    {
        const cv::Mat pixels(2, 4, type, cv::Scalar(10, 20, 30, 0));
        std::vector<unsigned char> encoded;
        ASSERT_TRUE(cv::imencode(".png", pixels, encoded));
        const PngImage png{4, 2, std::make_shared<const std::string>(encoded.begin(), encoded.end())};
        const RawImage raw = solidImage(4, 2, type == CV_8UC1 ? PixelFormat::Mono8 : PixelFormat::BGR8Packed, false, 0);
        for (const int inputSize : {640, 960})
        {
            const auto actual = prepareImage(png, inputSize);
            const auto expected = prepareImage(raw, inputSize);
            EXPECT_EQ(actual.tensor, expected.tensor);
            EXPECT_FLOAT_EQ(actual.letterbox.scale, expected.letterbox.scale);
            EXPECT_EQ(actual.letterbox.padLeft, expected.letterbox.padLeft);
            EXPECT_EQ(actual.letterbox.padTop, expected.letterbox.padTop);
            EXPECT_EQ(actual.letterbox.originalWidth, 4);
            EXPECT_EQ(actual.letterbox.originalHeight, 2);
        }
    }
}

TEST(ImagePipelineTest, PngRejectsCorruptTruncatedMismatchedOversizedAnd16BitInputs)
{
    std::vector<unsigned char> encoded;
    ASSERT_TRUE(cv::imencode(".png", cv::Mat(2, 4, CV_8UC3, cv::Scalar(10, 20, 30)), encoded));
    const std::string bytes(encoded.begin(), encoded.end());
    for (const auto& invalid : {std::string{}, std::string("not-png"), bytes.substr(0, 20), bytes.substr(0, 33)})
    {
        EXPECT_THROW(prepareImage(PngImage{4, 2, std::make_shared<const std::string>(invalid)}, 640), InvalidInferenceRequest);
    }
    EXPECT_THROW(prepareImage(PngImage{3, 2, std::make_shared<const std::string>(bytes)}, 640), InvalidInferenceRequest);
    std::string oversized = bytes;
    for (const std::size_t offset : {16U, 20U})
    {
        oversized[offset] = 0;
        oversized[offset + 1] = 0;
        oversized[offset + 2] = '\x20';
        oversized[offset + 3] = 0;
    }
    EXPECT_THROW(prepareImage(PngImage{8192, 8192, std::make_shared<const std::string>(oversized)}, 640), InvalidInferenceRequest);
    ASSERT_TRUE(cv::imencode(".png", cv::Mat(2, 4, CV_16UC1, cv::Scalar(1024)), encoded));
    EXPECT_THROW(prepareImage(PngImage{4, 2, std::make_shared<const std::string>(encoded.begin(), encoded.end())}, 640), InvalidInferenceRequest);
    EXPECT_THROW(prepareImage(PngImage{4, 2, std::make_shared<const std::string>(bytes)}, 800), std::invalid_argument);
}

TEST(ImagePipelineTest, OddPaddingPlacesExtraPixelOnBottom)
{
    const PreparedImage prepared = prepareImage(solidImage(640, 321, PixelFormat::BGR8Packed, false, 0), 640);
    EXPECT_EQ(prepared.letterbox.padTop, 159);
    EXPECT_FLOAT_EQ(prepared.tensor[158U * 640U], 114.0F / 255.0F);
    EXPECT_FLOAT_EQ(prepared.tensor[159U * 640U], 30.0F / 255.0F);
    EXPECT_FLOAT_EQ(prepared.tensor[479U * 640U], 30.0F / 255.0F);
    EXPECT_FLOAT_EQ(prepared.tensor[480U * 640U], 114.0F / 255.0F);
    EXPECT_EQ(prepareImage(solidImage(2, 4, PixelFormat::Mono8, true, 2), 960).letterbox.padLeft, 240);
}

TEST(ImagePipelineTest, RestoresCoordinatesFiltersThresholdAndKeepsOverlappingBoxes)
{
    std::vector<float> output(300 * 6, 0);
    const std::vector<float> rows{
        100, 120, 200, 220, 0.5F, 7,
        100, 120, 200, 220, 0.6F, 7,
        -100, -100, 10000, 10000, 0.8F, 2,
        0, 0, 1, 1, 0.49F, 0,
        0, 0, 1, 1, 0.9F, 0
    };
    std::copy(rows.begin(), rows.end(), output.begin());
    const auto detections = decodeDetections(output, LetterboxInfo{2, 20, 40, 200, 100}, 0.5F);
    ASSERT_EQ(detections.size(), 3U);
    EXPECT_FLOAT_EQ(detections[0].x1, 40);
    EXPECT_FLOAT_EQ(detections[0].y1, 40);
    EXPECT_FLOAT_EQ(detections[0].x2, 90);
    EXPECT_FLOAT_EQ(detections[0].y2, 90);
    EXPECT_EQ(detections[1].classId, 7);
    EXPECT_FLOAT_EQ(detections[2].x1, 0);
    EXPECT_FLOAT_EQ(detections[2].y1, 0);
    EXPECT_FLOAT_EQ(detections[2].x2, 199);
    EXPECT_FLOAT_EQ(detections[2].y2, 99);
}

TEST(ImagePipelineTest, EmptyOutputRemainsSuccessfulAndMalformedOutputIsRejected)
{
    std::vector<float> output(300 * 6, 0);
    const LetterboxInfo info{1, 0, 0, 640, 640};
    EXPECT_TRUE(decodeDetections(output, info, 0.25F).empty());
    EXPECT_THROW(decodeDetections(std::vector<float>(6), info, 0.25F), std::runtime_error);
    EXPECT_THROW(decodeDetections(output, info, std::numeric_limits<float>::quiet_NaN()), std::invalid_argument);
    EXPECT_THROW(prepareImage(solidImage(2, 2, PixelFormat::Mono8, false, 0), 800), std::invalid_argument);
    output[0] = std::numeric_limits<float>::infinity();
    output[2] = 100;
    output[3] = 100;
    output[4] = 0.9F;
    EXPECT_THROW(decodeDetections(output, info, 0.25F), std::runtime_error);
    output[0] = 0;
    for (float invalidClass : {-1.0F, 0.5F, std::numeric_limits<float>::quiet_NaN()})
    {
        output[5] = invalidClass;
        EXPECT_THROW(decodeDetections(output, info, 0.25F), std::runtime_error);
    }
}

} // namespace
