#include <chrono>
#include "imgui_tools.h"



std::uint64_t ProgressThread::getCaptureTimestamp()
{
	auto now = std::chrono::system_clock::now();
	uint64_t dis_millseconds = std::chrono::duration_cast<std::chrono::milliseconds>(now.time_since_epoch()).count()
		- std::chrono::duration_cast<std::chrono::seconds>(now.time_since_epoch()).count() * 1000;
	return dis_millseconds;
}