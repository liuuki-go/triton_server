#pragma once

#include <httplib.h>

#include <condition_variable>
#include <cstddef>
#include <deque>
#include <functional>
#include <mutex>
#include <thread>
#include <vector>

namespace triton_backend
{

class BoundedHttpTaskQueue final : public httplib::TaskQueue
{
public:
    BoundedHttpTaskQueue(std::size_t workerCount, std::size_t queueCapacity);
    void enqueue(std::function<void()> task) override;
    void shutdown() override;

private:
    void runWorker();

    const std::size_t queueCapacity_;
    std::vector<std::thread> workers_;
    std::deque<std::function<void()>> tasks_;
    bool stopping_{};
    std::mutex mutex_;
    std::condition_variable taskAvailable_;
};

} // namespace triton_backend
