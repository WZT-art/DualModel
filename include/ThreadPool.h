#pragma once

#include <thread>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <functional>
#include <atomic>
#include "AsyncLogger.h"

using ushort = unsigned short;

class Task {
public:
    Task() = default;
    explicit Task(std::function<void()>func) : func_(std::move(func)) {}

    void exec() {
        if(func_)
            func_();
    }
private:
    std::function<void()> func_;
};


class ThreadPool {
public:
    ThreadPool(ushort numThreads = 4) : numThreads_(numThreads) {
        running_.store(false, std::memory_order_release);
    }

    void start() {
        running_.store(true, std::memory_order_release);

        for(ushort i=0; i<numThreads_; i++) 
            workers_.emplace_back(&ThreadPool::worker, this);
    }

    template<typename F, typename... Args>
    void addTask(F&& f, Args&&... args) {
        auto task = std::bind(std::forward<F>(f), std::forward<Args>(args)...);

        queueMutex_.lock();
        taskQueue_.emplace(Task([task]() { task() }));
        queueMutex_.unlock();

        LOG_TRACE("taskQueue_ push a task, current task num: ", taskQueue_.size());
        queueCV_.notify_one();
    }

    void stop() {
        running_.store(false, std::memory_order_release);
        queueCV_.notify_all();

        for(auto& worker : workers_) {
            if(worker.joinable())
                worker.join();
        }
        workers_.clear();

        {
            std::lock_guard<std::mutex> lock(queueMutex_);
            while(!taskQueue_.empty())
                taskQueue_.pop();
        }
        LOG_TRACE("ThreadPool stoped");
    }

private:
    void worker() {
        while(running_.load(std::memory_order_acquire)) {
            LOG_TRACE("Thread ", std::this_thread::get_id(), " start");

            Task task;
            {
                std::unique_lock<std::mutex> lock(queueMutex_);
                queueCV_.wait(lock, [this]() -> bool {
                    return !taskQueue_.empty() || !running_.load(std::memory_order_acquire);
                });

                if(taskQueue_.empty() && !running_.load(std::memory_order_acquire)) {
                    LOG_TRACE("Worker thread ", std::this_thread::get_id(), " exit");
                    return;
                }

                if(!taskQueue_.empty()) {
                    task = std::move(taskQueue_.front());
                    taskQueue_.pop();
                }
            }
            task.exec();
        }
    }

    ushort numThreads_;
    std::vector<std::thread> workers_;
    std::queue<Task> taskQueue_;
    std::mutex queueMutex_;
    std::condition_variable queueCV_;
    std::atomic<bool> running_;
};