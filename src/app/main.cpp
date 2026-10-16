#include "configuration/BackendConfig.h"
#include "api/BackendHttpServer.h"
#include "services/impl/BackendService.h"

#include <pthread.h>
#include <signal.h>

#include <atomic>
#include <iostream>
#include <filesystem>
#include <stdexcept>
#include <thread>

#ifndef DEFAULT_CONFIG_PATH
#define DEFAULT_CONFIG_PATH "config/server.yaml"
#endif

namespace
{

std::string defaultConfigPath()
{
    std::error_code error;
    const auto executable = std::filesystem::read_symlink("/proc/self/exe", error);
    if (!error)
    {
        const auto deployed = executable.parent_path().parent_path() / "config/server.yaml";
        if (std::filesystem::is_regular_file(deployed)) return deployed.string();
    }
    return DEFAULT_CONFIG_PATH;
}

} // namespace

int main(int argc, char* argv[])
{
    try
    {
        std::string configPath = defaultConfigPath();
        triton_backend::ModelSelection selection;
        bool modelIdProvided = false;
        bool modelVersionProvided = false;
        for (int index = 1; index < argc; ++index)
        {
            const std::string argument = argv[index];
            if ((argument == "--config" || argument == "--model-id" || argument == "--model-version") && index + 1 < argc)
            {
                const std::string value = argv[++index];
                if (argument == "--config") configPath = value;
                else if (argument == "--model-id") { selection.modelId = value; modelIdProvided = true; }
                else { selection.version = value; modelVersionProvided = true; }
            }
            else if (argc == 2 && argument.rfind("--", 0) != 0) configPath = argument;
            else throw std::invalid_argument("用法：triton_backend [--config server.yaml] [--model-id ID --model-version VERSION]");
        }
        if (modelIdProvided != modelVersionProvided || (modelIdProvided && (selection.modelId.empty() || selection.version.empty())))
            throw std::invalid_argument("--model-id 和 --model-version 必须同时提供且非空");
        const auto config = triton_backend::loadBackendConfig(configPath);
        sigset_t signals;
        sigemptyset(&signals);
        sigaddset(&signals, SIGINT);
        sigaddset(&signals, SIGTERM);
        if (pthread_sigmask(SIG_BLOCK, &signals, nullptr) != 0)
            throw std::runtime_error("无法设置终止信号处理");
        triton_backend::BackendService service(config.runtime, config.models);
        if (!selection.modelId.empty()) service.selectModel(selection);
        triton_backend::BackendHttpServer server(service, config);
        const auto port = server.bind();
        if (port < 0) throw std::runtime_error("无法绑定 HTTP 端口");
        std::atomic<bool> finished{false};
        std::atomic<bool> shutdownRequested{false};
        std::thread monitor([&]
        {
            while (!finished.load())
            {
                const timespec timeout{0, 100000000};
                const auto signal = sigtimedwait(&signals, nullptr, &timeout);
                if (signal == SIGINT || signal == SIGTERM) shutdownRequested = true;
                if (shutdownRequested.load()) server.stop();
            }
        });
        if (config.autoStart && selection.modelId.empty())
        {
            try { service.selectModel({config.models.front().id, config.models.front().externalVersion}); }
            catch (const std::exception& error) { std::cerr << "模型加载失败：" << error.what() << '\n'; }
        }
        std::cout << "C++ / Triton 服务：http://" << config.server.host << ':' << port << std::endl;
        bool succeeded = false;
        try { succeeded = server.listenAfterBind(); }
        catch (...) { finished = true; monitor.join(); throw; }
        finished = true;
        monitor.join();
        service.stop();
        if (!succeeded && !shutdownRequested.load()) throw std::runtime_error("HTTP 监听异常退出");
        return 0;
    }
    catch (const InvalidInferenceRequest& error)
    {
        std::cerr << "后端失败 [" << error.code() << "]：" << error.what() << '\n';
        return 1;
    }
    catch (const std::exception& error)
    {
        std::cerr << "后端失败：" << error.what() << '\n';
        return 1;
    }
}
