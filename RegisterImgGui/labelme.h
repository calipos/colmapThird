#ifndef _LABELME_H_
#define _LABELME_H_
#include <map>
#include <filesystem>
#include "Eigen/Core"
namespace labelme
{
	bool readPtsFromLabelMeJson(const std::filesystem::path& jsonPath,
		std::map<std::string, Eigen::Vector2d>& cornerInfo, Eigen::Vector2i& imgSizeWH, std::string* imgpath = nullptr);
}



#endif // !_LABELME_H_
