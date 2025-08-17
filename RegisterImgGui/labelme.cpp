#include <fstream>
#include <map>
#include "labelme.h"
#include "json/json.h"
#include "log.h"
#include "opencv2/opencv.hpp"
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


	int writeLabelMeLinestripJson(const std::filesystem::path& imgPath, const std::map<std::string, Eigen::Vector2d>& sortedPtsBaseLabel)
	{
		std::filesystem::path root = imgPath.parent_path();
		std::string stem = imgPath.stem().string();
		std::filesystem::path jsonPath = root / (stem + ".json");
		cv::Mat image = cv::imread(imgPath.string());
		//std::string encodedImage = Base64::encodeMat(image, ".jpg");
		Json::Value labelRoot;
		labelRoot["version"] = Json::Value("5.4.1");
		labelRoot["flags"] = Json::Value(Json::objectValue);

		Json::Value labelPts;
		for (const auto& d : sortedPtsBaseLabel)
		{
			Json::Value labelPt;
			labelPt["label"] = Json::Value(d.first);
			{
				Json::Value pt;
				pt.append(d.second[0]);
				pt.append(d.second[1]);
				labelPt["points"].append(pt);
			}
			labelPt["group_id"] = Json::nullValue;
			labelPt["description"] = Json::Value("");
			labelPt["shape_type"] = Json::Value("point");
			labelPt["flags"] = Json::Value(Json::objectValue);
			labelPt["mask"] = Json::nullValue;
			labelRoot["shapes"].append(labelPt);
		}
		labelRoot["imagePath"] = Json::Value(imgPath.filename().string());
		labelRoot["imageData"] = Json::Value();
		labelRoot["imageHeight"] = Json::Value(image.rows);
		labelRoot["imageWidth"] = Json::Value(image.cols);

		Json::StyledWriter sw;
		std::fstream fout(jsonPath, std::ios::out);
		fout << sw.write(labelRoot);
		fout.close();
		return 0;
	}
}

int test_writelabel()
{
	std::map<std::string, Eigen::Vector2d>  sortedPtsBaseLabel = { {"a",{300,240}}, {"b",{700,640}} };
	labelme::writeLabelMeLinestripJson("D:/repo/colmap-third/a.bmp", sortedPtsBaseLabel);
	return 0;
}