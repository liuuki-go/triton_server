#include <nlohmann/json.hpp>
#include <opencv2/imgcodecs.hpp>

#include <algorithm>
#include <cctype>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace
{

namespace fs = std::filesystem;
using Json = nlohmann::json;

bool isSupportedImage(const fs::path& path)
{
    std::string extension = path.extension().string();
    std::transform(extension.begin(), extension.end(), extension.begin(),
        [](unsigned char value) { return static_cast<char>(std::tolower(value)); });
    return extension == ".jpg" || extension == ".jpeg" || extension == ".bmp" || extension == ".png";
}

std::vector<fs::path> collectImages(const fs::path& source)
{
    std::vector<fs::path> images;
    for (const auto& entry : fs::recursive_directory_iterator(source))
    {
        if (entry.is_regular_file() && isSupportedImage(entry.path())) images.push_back(entry.path());
    }
    std::sort(images.begin(), images.end());
    if (images.empty()) throw std::runtime_error("输入目录没有支持的图像");
    return images;
}

void prepareDataset(const fs::path& source, const fs::path& cache)
{
    const auto relativeCache = cache.lexically_relative(source);
    if (!relativeCache.empty() && *relativeCache.begin() != "..")
        throw std::invalid_argument("输出目录必须位于输入目录之外");
    if (fs::exists(cache) && (!fs::is_directory(cache) || !fs::is_empty(cache)))
        throw std::invalid_argument("输出目录必须不存在或为空，不覆盖已有文件");

    const auto images = collectImages(source);
    fs::create_directories(cache);
    Json descriptions = Json::array();
    std::map<std::pair<int, int>, std::size_t> sizes;
    std::uintmax_t sourceBytes = 0;
    std::uintmax_t pngBytes = 0;
    for (const auto& path : images)
    {
        const auto relativeSource = path.lexically_relative(source);
        const auto relativePng = fs::path("images") / fs::path(relativeSource.generic_string() + ".png");
        const auto destination = cache / relativePng;
        if (fs::exists(destination)) throw std::runtime_error("输出文件已经存在：" + destination.string());
        const cv::Mat image = cv::imread(path.string(), cv::IMREAD_COLOR | cv::IMREAD_IGNORE_ORIENTATION);
        if (image.empty()) throw std::runtime_error("图像解码失败：" + path.string());
        fs::create_directories(destination.parent_path());
        if (!cv::imwrite(destination.string(), image, {cv::IMWRITE_PNG_COMPRESSION, 3}))
            throw std::runtime_error("PNG 写入失败：" + destination.string());
        const auto inputSize = fs::file_size(path);
        const auto outputSize = fs::file_size(destination);
        descriptions.push_back(Json{{"sourceRelativePath", relativeSource.generic_string()},
            {"sourceExtension", path.extension().string()}, {"width", image.cols}, {"height", image.rows},
            {"pngRelativePath", relativePng.generic_string()}, {"sourceBytes", inputSize}, {"pngBytes", outputSize}});
        ++sizes[{image.cols, image.rows}];
        sourceBytes += inputSize;
        pngBytes += outputSize;
    }
    Json distribution = Json::array();
    for (const auto& [size, count] : sizes)
    {
        distribution.push_back(Json{{"width", size.first}, {"height", size.second}, {"count", count}});
    }
    const Json summary{{"imageCount", images.size()}, {"sourceBytes", sourceBytes},
        {"pngBytes", pngBytes}, {"sizeDistribution", distribution}};
    const Json manifest{{"sourceDirectory", source.generic_string()}, {"cacheDirectory", cache.generic_string()},
        {"pngCompression", 3}, {"images", descriptions}, {"summary", summary}};
    const auto manifestPath = cache / "dataset.json";
    if (fs::exists(manifestPath)) throw std::runtime_error("清单文件已经存在：" + manifestPath.string());
    std::ofstream output(manifestPath, std::ios::binary);
    if (!output) throw std::runtime_error("无法创建清单：" + manifestPath.string());
    output << manifest.dump(2) << '\n';
    output.close();
    if (!output) throw std::runtime_error("清单写入失败：" + manifestPath.string());
    std::cout << summary.dump(2) << '\n';
}

} // namespace

int main(int argc, char* argv[])
{
    try
    {
        if (argc != 3) throw std::invalid_argument("用法：prepare_dataset <输入图像目录> <空的 PNG 缓存目录>");
        const auto source = fs::canonical(argv[1]);
        if (!fs::is_directory(source)) throw std::invalid_argument("输入必须是目录");
        const auto cache = fs::weakly_canonical(argv[2]);
        prepareDataset(source, cache);
        return 0;
    }
    catch (const std::exception& error)
    {
        std::cerr << "准备图像数据失败：" << error.what() << '\n';
        return 1;
    }
}
