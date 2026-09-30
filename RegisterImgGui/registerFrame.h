#ifndef _REGIST_FRAME_H_
#define _REGIST_FRAME_H_
#include <string>
#include <vector>
bool registFrame(bool* show_far_align_window = nullptr);
int register_incremental(const std::string& folder);
int register_incremental_loop(const std::string& folder);
int register_incremental_base_hint(const std::string& folder, const std::vector<std::string>& initViewName, const std::vector<std::string>& discardViewName, const bool& optimCameraIntr, const bool& saveResult);
#endif // !_REGIST_FRAME_H_
