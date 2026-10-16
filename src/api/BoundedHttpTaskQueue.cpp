#include "api/BoundedHttpTaskQueue.h"

#include <utility>

namespace triton_backend
{

BoundedHttpTaskQueue::BoundedHttpTaskQueue(std::size_t workerCount, std::size_t queueCapacity)
    : queueCapacity_(queueCapacity)
{
    try
    {
        workers_.reserve(workerCount);
        for (std::size_t index = 0; index < workerCount; ++index)
        {
            workers_.emplace_back([this] { runWorker(); });
        }
    }
    catch (...)
    {
        shutdown();
        throw;
    }
}

void BoundedHttpTaskQueue::enqueue(std::function<void()> task)
{
    {
        std::lock_guard<std::mutex> lock(mutex_);
        if (tasks_.size() < queueCapacity_)
        {
            tasks_.push_back(std::move(task));
            taskAvailable_.notify_one();
            return;
        }
    }
    // 保留接入线程处理溢出任务的行为，队列锁不跨越业务请求。
    task();
}

void BoundedHttpTaskQueue::shutdown()
{
    {
        std::lock_guard<std::mutex> lock(mutex_);
        stopping_ = true;
    }
    taskAvailable_.notify_all();
    for (std::thread& worker : workers_)
    {
        if (worker.joinable()) worker.join();
    }
}

void BoundedHttpTaskQueue::runWorker()
{
    while (true)
    {
        std::function<void()> task;
        {
            std::unique_lock<std::mutex> lock(mutex_);
            taskAvailable_.wait(lock, [this] { return stopping_ || !tasks_.empty(); });
            // 退出前排空已经接收的任务。
            if (stopping_ && tasks_.empty()) return;
            task = std::move(tasks_.front());
            tasks_.pop_front();
        }
        task();
    }
}

} // namespace triton_backend
