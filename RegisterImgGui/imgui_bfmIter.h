#ifndef _BFMITER_FRAME_H_
#define _BFMITER_FRAME_H_
#include <string>
#include <filesystem>
#include <string>
#include "Eigen/Core"
#include "opencv2/opencv.hpp"
#include "imgui.h" 
#include "bfm.h"
#include "meshDraw.h"
bool bfmIterFrame(bool* show_bfmIter_window = nullptr);

enum class IterType
{};
class BfmIter
{
public:
	bfm::Bfm2019* bfmIns{ nullptr };
	BfmIter() = default;
	BfmIter(const  std::filesystem::path& mvsResultDir, const  std::filesystem::path& modelDirPath);
	~BfmIter();
	bool iter(const std::vector<cv::Point3f>& src, const std::vector<cv::Point3f>& tar, const IterType& type);
	Eigen::Matrix3f bfm_R;
	Eigen::RowVector3f bfm_t;
	std::vector<cv::Mat>imgs;
	std::vector<cv::Mat>renders;
	std::vector<cv::Mat>renderPts;
	std::vector<cv::Mat>renderMasks;
	std::vector<ImVec2>shifts;;
	std::vector<std::filesystem::path>imgPaths;
	std::vector<std::string>imgNameForlist;
	std::filesystem::path imgDirPath_;
	std::filesystem::path modelDirPath_;
	static GLuint image_texture;
	static int viewWindowHeight;
	static int viewWindowWidth;
	static int imgPickIdx;
	meshdraw::Mesh msh;
	std::vector<meshdraw::Camera> imgCameras;
};
#endif // !_ANNOTATION_FRAME_H_
