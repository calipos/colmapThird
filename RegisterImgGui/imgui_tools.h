#ifndef _IMGUI_TOOLS_H__
#define _IMGUI_TOOLS_H__
#include <thread>
#include <vector>
#include <string>
#include <atomic>
#include "imgui.h"
#include "opencv2/opencv.hpp"
#define GL_SILENCE_DEPRECATION 
#include <GLFW/glfw3.h> // Will drag system OpenGL headers canvas
bool LoadTextureFromMat(const cv::Mat& originalImg, GLuint* out_texture, int* out_width, int* out_height);
void listComponent(const std::string& listName, const ImVec2& wh, std::vector<std::string>& itemNames, int& pickIdxOut, bool& changedRightNow);
struct ProgressThread
{
	std::atomic<int> procRunning{ 0 };
	std::atomic<int> denominator{ 0 };
	std::atomic<int> numerator{ 0 };
	std::thread* proc{ nullptr };
	std::string msg{ "" };
	static std::uint64_t getCaptureTimestamp();
};

cv::Vec3b getColor();
ImU32 getImguiColor();
#endif // !_IMGUI_TOOLS_H__
