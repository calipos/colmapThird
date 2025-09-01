#ifndef _LABELME_H_
#define _LABELME_H_
#include <map>
#include <filesystem>
#include "Eigen/Core"
#include <Eigen/Geometry>
namespace labelme
{
	bool readPtsFromLabelMeJson(const std::filesystem::path& jsonPath,
		std::map<std::string, Eigen::Vector2d>& cornerInfo, Eigen::Vector2i& imgSizeWH, std::string* imgpath = nullptr);
	int writeLabelMeLinestripJson(const std::filesystem::path& imgPath, const std::map<std::string, Eigen::Vector2d>& sortedPtsBaseLabel);


	bool readCmaeraFromRegisterJson(const std::filesystem::path& imgJsonPath, Eigen::Matrix4d& cameraMatrix, Eigen::Matrix4d& Rt);
	Eigen::MatrixX2d readPtsFromRegisterJson(const std::filesystem::path& imgJsonPath);
}



#endif // !_LABELME_H_
