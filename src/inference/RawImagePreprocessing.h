#pragma once

#include "domain/InferenceTypes.h"

#include <opencv2/core/mat.hpp>

// BGR TopDown返回只读借用视图，调用方必须保留image.data且不得修改视图。
// 其他格式或BottomUp返回独立Mat；所有返回值都是TopDown BGR三通道。
cv::Mat rawImageToBgr(const RawImage& image);
