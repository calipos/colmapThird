#include "threading.h"

#include "log.h"


Thread::Thread()
    : started_(false),
    stopped_(false),
    paused_(false),
    pausing_(false),
    finished_(false),
    setup_(false),
    setup_valid_(false) {
    RegisterCallback(STARTED_CALLBACK);
    RegisterCallback(FINISHED_CALLBACK);
}

void Thread::Start() {
    std::unique_lock<std::mutex> lock(mutex_);
    if (!started_ || finished_)
    {

    }
    else
    {
        LOG_ERR_OUT;
    }
    Wait();
    timer_.Restart();
    thread_ = std::thread(&Thread::RunFunc, this);
    started_ = true;
    stopped_ = false;
    paused_ = false;
    pausing_ = false;
    finished_ = false;
    setup_ = false;
    setup_valid_ = false;
}

void Thread::Stop() {
    {
        std::unique_lock<std::mutex> lock(mutex_);
        stopped_ = true;
    }
    Resume();
}

void Thread::Pause() {
    std::unique_lock<std::mutex> lock(mutex_);
    paused_ = true;
}

void Thread::Resume() {
    std::unique_lock<std::mutex> lock(mutex_);
    if (paused_) {
        paused_ = false;
        pause_condition_.notify_all();
    }
}

void Thread::Wait() {
    if (thread_.joinable()) {
        thread_.join();
    }
}

bool Thread::IsStarted() {
    std::unique_lock<std::mutex> lock(mutex_);
    return started_;
}

bool Thread::IsStopped() {
    std::unique_lock<std::mutex> lock(mutex_);
    return stopped_;
}

bool Thread::IsPaused() {
    std::unique_lock<std::mutex> lock(mutex_);
    return paused_;
}

bool Thread::IsRunning() {
    std::unique_lock<std::mutex> lock(mutex_);
    return started_ && !pausing_ && !finished_;
}

bool Thread::IsFinished() {
    std::unique_lock<std::mutex> lock(mutex_);
    return finished_;
}

void Thread::AddCallback(const int id, std::function<void()> func) {
    if (!func)
    {
        LOG_ERR_OUT;
    }
    if (callbacks_.count(id)== 0)
    {
        LOG_ERR_OUT;
    }
    callbacks_.at(id).push_back(std::move(func));
}

void Thread::RegisterCallback(const int id) {
    callbacks_.emplace(id, std::list<std::function<void()>>());
}

void Thread::Callback(const int id) const {
    if (callbacks_.count(id) == 0)
    {
        LOG_ERR_OUT;
    }
    for (const auto& callback : callbacks_.at(id)) {
        callback();
    }
}

std::thread::id Thread::GetThreadId() const {
    return std::this_thread::get_id();
}

void Thread::SignalValidSetup() {
    std::unique_lock<std::mutex> lock(mutex_);
    if (setup_)
    {
        LOG_ERR_OUT;
    }
    setup_ = true;
    setup_valid_ = true;
    setup_condition_.notify_all();
}

void Thread::SignalInvalidSetup() {
    std::unique_lock<std::mutex> lock(mutex_);
    if (setup_)
    {
        LOG_ERR_OUT;
    }
    setup_ = true;
    setup_valid_ = false;
    setup_condition_.notify_all();
}

const class Timer& Thread::GetTimer() const { return timer_; }

void Thread::BlockIfPaused() {
    std::unique_lock<std::mutex> lock(mutex_);
    if (paused_) {
        pausing_ = true;
        timer_.Pause();
        pause_condition_.wait(lock);
        pausing_ = false;
        timer_.Resume();
    }
}

bool Thread::CheckValidSetup() {
    std::unique_lock<std::mutex> lock(mutex_);
    if (!setup_) {
        setup_condition_.wait(lock);
    }
    return setup_valid_;
}

void Thread::RunFunc() {
    Callback(STARTED_CALLBACK);
    Run();
    {
        std::unique_lock<std::mutex> lock(mutex_);
        finished_ = true;
        timer_.Pause();
    }
    Callback(FINISHED_CALLBACK);
}

ThreadPool::ThreadPool(const int num_threads)
    : stopped_(false), num_active_workers_(0) {
    const int num_effective_threads = GetEffectiveNumThreads(num_threads);
    for (int index = 0; index < num_effective_threads; ++index) {
        std::function<void(void)> worker =
            std::bind(&ThreadPool::WorkerFunc, this, index);
        workers_.emplace_back(worker);
    }
}

ThreadPool::~ThreadPool() { Stop(); }

void ThreadPool::Stop() {
    {
        std::unique_lock<std::mutex> lock(mutex_);

        if (stopped_) {
            return;
        }

        stopped_ = true;

        std::queue<std::function<void()>> empty_tasks;
        std::swap(tasks_, empty_tasks);
    }

    task_condition_.notify_all();

    for (auto& worker : workers_) {
        worker.join();
    }

    finished_condition_.notify_all();
}

void ThreadPool::Wait() {
    std::unique_lock<std::mutex> lock(mutex_);
    if (!tasks_.empty() || num_active_workers_ > 0) {
        finished_condition_.wait(
            lock, [this]() { return tasks_.empty() && num_active_workers_ == 0; });
    }
}

void ThreadPool::WorkerFunc(const int index) {
    {
        std::lock_guard<std::mutex> lock(mutex_);
        thread_id_to_index_.emplace(GetThreadId(), index);
    }

    while (true) {
        std::function<void()> task;
        {
            std::unique_lock<std::mutex> lock(mutex_);
            task_condition_.wait(lock,
                [this] { return stopped_ || !tasks_.empty(); });
            if (stopped_ && tasks_.empty()) {
                return;
            }
            task = std::move(tasks_.front());
            tasks_.pop();
            num_active_workers_ += 1;
        }

        task();

        {
            std::unique_lock<std::mutex> lock(mutex_);
            num_active_workers_ -= 1;
        }

        finished_condition_.notify_all();
    }
}

std::thread::id ThreadPool::GetThreadId() const {
    return std::this_thread::get_id();
}

int ThreadPool::GetThreadIndex() {
    std::unique_lock<std::mutex> lock(mutex_);
    return thread_id_to_index_.at(GetThreadId());
}

int GetEffectiveNumThreads(const int num_threads) {
    int num_effective_threads = num_threads;
    if (num_threads <= 0) {
        num_effective_threads = std::thread::hardware_concurrency();
    }

    if (num_effective_threads <= 0) {
        num_effective_threads = 1;
    }

    return num_effective_threads;
}
