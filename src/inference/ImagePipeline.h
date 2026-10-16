#pragma once

#include "domain/InferenceTypes.h"

#include <vector>

namespace triton_backend
{

struct LetterboxInfo
{
    float scale{};
    int padLeft{};
    int padTop{};
    int originalWidth{};
    int originalHeight{};
};

struct PreparedImage
{
    std::vector<float> tensor;
    LetterboxInfo letterbox;
};

PreparedImage prepareImage(const RawImage& image, int inputSize);
PreparedImage prepareImage(const PngImage& image, int inputSize);

std::vector<Detection> decodeDetections(
    const std::vector<float>& output,
    const LetterboxInfo& letterbox,
    float confidenceThreshold
);

} // namespace triton_backend
