#ifndef _TIMER_H_
#define _TIMER_H_
#include <chrono>

class Timer {
public:
	Timer();

	void Start();
	void Restart();
	void Pause();
	void Resume();
	void Reset();

	double ElapsedMicroSeconds() const;
	double ElapsedSeconds() const;
	double ElapsedMinutes() const;
	double ElapsedHours() const;

	void PrintSeconds() const;
	void PrintMinutes() const;
	void PrintHours() const;

private:
	bool started_;
	bool paused_;
	std::chrono::high_resolution_clock::time_point start_time_;
	std::chrono::high_resolution_clock::time_point pause_time_;
};
#endif // !_TIMER_H_
