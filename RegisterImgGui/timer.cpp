#include "timer.h"
#include "log.h"
#include "misc.h"
#include "colString.h"
Timer::Timer() : started_(false), paused_(false) {}

void Timer::Start() {
    started_ = true;
    paused_ = false;
    start_time_ = std::chrono::high_resolution_clock::now();
}

void Timer::Restart() {
    started_ = false;
    Start();
}

void Timer::Pause() {
    paused_ = true;
    pause_time_ = std::chrono::high_resolution_clock::now();
}

void Timer::Resume() {
    paused_ = false;
    start_time_ += std::chrono::high_resolution_clock::now() - pause_time_;
}

void Timer::Reset() {
    started_ = false;
    paused_ = false;
}

double Timer::ElapsedMicroSeconds() const {
    if (!started_) {
        return 0.0;
    }
    if (paused_) {
        return std::chrono::duration_cast<std::chrono::microseconds>(pause_time_ -
            start_time_)
            .count();
    }
    else {
        return std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::high_resolution_clock::now() - start_time_)
            .count();
    }
}

double Timer::ElapsedSeconds() const { return ElapsedMicroSeconds() / 1e6; }

double Timer::ElapsedMinutes() const { return ElapsedSeconds() / 60; }

double Timer::ElapsedHours() const { return ElapsedMinutes() / 60; }

void Timer::PrintSeconds() const {
    LOG_OUT << StringPrintf("Elapsed time: %.5f [seconds]", ElapsedSeconds());
}

void Timer::PrintMinutes() const {
    LOG_OUT << StringPrintf("Elapsed time: %.3f [minutes]", ElapsedMinutes());
}

void Timer::PrintHours() const {
    LOG_OUT << StringPrintf("Elapsed time: %.3f [hours]", ElapsedHours());
}