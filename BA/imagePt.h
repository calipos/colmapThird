#ifndef _IMAGE_POINT_H_
#define _IMAGE_POINT_H_
#include <filesystem>
#include <string>
#include <vector>
#include <array>
#include <map>
namespace col
{
	struct ImgPt
	{
		int imgId;
		std::array<double, 4>wxyz;
		std::array<double, 3>r;
		std::array<double, 3>t;
		int cameraId;
		std::filesystem::path imgPath;
		std::vector<std::array<double, 2>>imgPts; 
		ImgPt();
		ImgPt(const int& imgId_,
			const std::array<double, 4>& wxyz,
			const std::array<double, 3>&t,
			const int& cameraId,
			const std::filesystem::path& imgPath,
			std::vector<std::array<double, 2>>&&imgPts_);
	};
}

std::vector<col::ImgPt> readImgPtsFromTXT(const std::filesystem::path& cameraTXT);
#endif // _IMAGE_POINT_H_
