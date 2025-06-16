#include <fstream>
#include "labelme.h"
#include "json/json.h"
#include "log.h"

namespace labelme
{
	bool readPtsFromLabelMeJson(const std::filesystem::path& jsonPath,
		std::map<std::string, Eigen::Vector2d>& cornerInfo, Eigen::Vector2i& imgSizeWH,std::string*imgpath)
	{
		cornerInfo.clear();
		std::stringstream ss;
		std::string aline;
		std::fstream fin(jsonPath, std::ios::in);
		while (std::getline(fin, aline))
		{
			ss << aline;
		}
		fin.close();
		aline = ss.str();
		JSONCPP_STRING err;
		Json::Value newRoot;
		const auto rawJsonLength = static_cast<int>(aline.length());
		Json::CharReaderBuilder newBuilder;
		const std::unique_ptr<Json::CharReader> newReader(newBuilder.newCharReader());
		if (!newReader->parse(aline.c_str(), aline.c_str() + rawJsonLength, &newRoot,
			&err)) {
			return  false;
		}
		auto newMemberNames = newRoot.getMemberNames();
		auto pathNode = newRoot["imagePath"];
		if (imgpath !=nullptr)
		{
			*imgpath = newRoot["imagePath"].asString();
		}
		auto shapeNode = newRoot["shapes"];
		if (pathNode.isNull() || shapeNode.isNull() || !shapeNode.isArray())
		{
			return  false;
		}
		imgSizeWH.y() = newRoot["imageHeight"].asInt();
		imgSizeWH.x() = newRoot["imageWidth"].asInt();
		for (int i = 0; i < shapeNode.size(); i++)
		{
			auto label = shapeNode[i];
			if (label["label"].isNull() || label["points"].isNull() || label["shape_type"].isNull())
			{
				return  false;
			}
			std::string shapeType = label["shape_type"].asString();
			if (shapeType.compare("point") != 0)
			{
				continue;
			}
			std::string cornerLabel = label["label"].asString();
			if (label["points"].size() == 0)
			{
				LOG_ERR_OUT << "not unique!";
				return false;
			}
			cornerInfo[cornerLabel].x() = label["points"][0][0].asDouble();
			cornerInfo[cornerLabel].y() = label["points"][0][1].asDouble();
		}
		if (cornerInfo.size() == 0)
		{
			return false;
		}
		return true;
	}
}

