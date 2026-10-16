#include "inference/RawImagePreprocessing.h"

#include <gtest/gtest.h>

#include <array>
#include <limits>
#include <memory>
#include <string>

namespace
{

TEST(RawImagePreprocessingTest, PreservesPixelsForFormatsStrideAndRowDirection)
{
    const std::array<cv::Vec3b, 4> colorPixels{
        cv::Vec3b{12, 34, 56}, cv::Vec3b{78, 90, 123},
        cv::Vec3b{201, 32, 67}, cv::Vec3b{111, 220, 21}
    };
    const std::array<unsigned char, 4> grayPixels{5, 35, 65, 250};
    for (const PixelFormat format : {
        PixelFormat::Mono8, PixelFormat::RGB8Packed, PixelFormat::BGR8Packed
    })
    {
        for (const bool bottomUp : {false, true})
        {
            for (const std::size_t padding : {0U, 3U})
            {
                SCOPED_TRACE("format=" + std::to_string(static_cast<int>(format)) +
                             " bottomUp=" + std::to_string(bottomUp) +
                             " padding=" + std::to_string(padding));
                const std::size_t channels = format == PixelFormat::Mono8 ? 1 : 3;
                const std::size_t stride = 2 * channels + padding;
                std::string bytes(stride * 2, '\xA5');
                for (int row = 0; row < 2; ++row)
                {
                    const int storageRow = bottomUp ? 1 - row : row;
                    for (int column = 0; column < 2; ++column)
                    {
                        const std::size_t index = row * 2 + column;
                        const std::size_t offset = storageRow * stride + column * channels;
                        if (format == PixelFormat::Mono8)
                        {
                            bytes[offset] = static_cast<char>(grayPixels[index]);
                        }
                        else
                        {
                            const cv::Vec3b& bgr = colorPixels[index];
                            bytes[offset] = static_cast<char>(format == PixelFormat::RGB8Packed ? bgr[2] : bgr[0]);
                            bytes[offset + 1] = static_cast<char>(bgr[1]);
                            bytes[offset + 2] = static_cast<char>(format == PixelFormat::RGB8Packed ? bgr[0] : bgr[2]);
                        }
                    }
                }
                const auto storage = std::make_shared<const std::string>(bytes);
                const RawImage image{2, 2, format, stride, bottomUp, storage};

                const cv::Mat result = rawImageToBgr(image);

                ASSERT_EQ(result.type(), CV_8UC3);
                ASSERT_EQ(result.cols, 2);
                ASSERT_EQ(result.rows, 2);
                for (int row = 0; row < 2; ++row)
                {
                    for (int column = 0; column < 2; ++column)
                    {
                        const std::size_t index = row * 2 + column;
                        const cv::Vec3b expected = format == PixelFormat::Mono8
                            ? cv::Vec3b{grayPixels[index], grayPixels[index], grayPixels[index]}
                            : colorPixels[index];
                        EXPECT_EQ(result.at<cv::Vec3b>(row, column), expected);
                    }
                }
                EXPECT_EQ(*storage, bytes);
                if (format == PixelFormat::BGR8Packed && !bottomUp)
                {
                    EXPECT_EQ(result.data, reinterpret_cast<const unsigned char*>(storage->data()));
                    EXPECT_EQ(result.step[0], stride);
                }
                else
                {
                    EXPECT_NE(result.data, reinterpret_cast<const unsigned char*>(storage->data()));
                }
            }
        }
    }
}

TEST(RawImagePreprocessingTest, RejectsInvalidLayoutBeforeWrappingMemory)
{
    const auto storage = std::make_shared<const std::string>(6, '\0');
    EXPECT_THROW(rawImageToBgr(RawImage{0, 1, PixelFormat::BGR8Packed, 6, false, storage}), std::invalid_argument);
    EXPECT_THROW(rawImageToBgr(RawImage{2, 1, PixelFormat::BGR8Packed, 5, false, storage}), std::invalid_argument);
    EXPECT_THROW(rawImageToBgr(RawImage{2, 1, static_cast<PixelFormat>(99), 6, false, storage}), std::invalid_argument);
    EXPECT_THROW(rawImageToBgr(RawImage{2, 1, PixelFormat::BGR8Packed, 6, false, nullptr}), std::invalid_argument);
    EXPECT_THROW(rawImageToBgr(RawImage{2, 2, PixelFormat::BGR8Packed, 6, false, storage}), std::invalid_argument);
    EXPECT_THROW(rawImageToBgr(RawImage{1, 2, PixelFormat::Mono8,
                 std::numeric_limits<std::size_t>::max(), false, storage}), std::invalid_argument);
}

} // namespace
