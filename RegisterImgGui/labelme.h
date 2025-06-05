#ifndef _LABELME_H_
#define _LABELME_H_
#include <filesystem>
#include "json/json.h"
#include "Eigen/Core"
namespace labelme
{
	bool readPtsFromLabelMeJson(const std::filesystem::path& jsonPath,
		std::map<std::string, Eigen::Vector2i>& cornerInfo, Eigen::Vector2i& imgSizeWH);
}



#endif // !_LABELME_H_
