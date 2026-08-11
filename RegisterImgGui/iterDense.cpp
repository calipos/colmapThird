#include <numeric>
#include <vector>
#include <filesystem>
#include <fstream>
#include <atomic>
#include <map>
#include "Eigen/Core"
#include <Eigen/Geometry>
#include <igl/ray_mesh_intersect.h>
#include "igl/readOBJ.h"
#include "opencv2/opencv.hpp"
#include "json/json.h"
#include "log.h"
#include "opencvTools.h"


bool readRsultJson(const std::filesystem::path& jsonPath, Eigen::Matrix4d& Rt, Eigen::Matrix4d& K,int&height,int&width, std::filesystem::path& imgPath)
{ 
	std::string aline;
	{
		std::fstream fin(jsonPath, std::ios::in);
		if (!fin.is_open())
		{
			LOG_ERR_OUT << "connot open:" << jsonPath;
			return false;
		}
		std::stringstream ss;
		ss << fin.rdbuf();
		aline = ss.str();
	} 
	JSONCPP_STRING err;
	Json::Value newRoot;
	const auto rawJsonLength = static_cast<int>(aline.length());
	Json::CharReaderBuilder newBuilder;
	const std::unique_ptr<Json::CharReader> newReader(newBuilder.newCharReader());
	if (!newReader->parse(aline.c_str(), aline.c_str() + rawJsonLength, &newRoot,
		&err)) {
		return  false;
	} 

	if (!newRoot.isObject()
		|| !newRoot.isMember("Qt")
		|| !newRoot.isMember("cx")
		|| !newRoot.isMember("cy")
		|| !newRoot.isMember("fx")
		|| !newRoot.isMember("fy")
		|| !newRoot.isMember("height")
		|| !newRoot.isMember("width")
		|| !newRoot.isMember("imagePath"))
	{
		return false;
	}

	auto QtNode = newRoot["Qt"];
	auto cxNode = newRoot["cx"];
	auto cyNode = newRoot["cy"];
	auto fxNode = newRoot["fx"];
	auto fyNode = newRoot["fy"];
	auto heightNode = newRoot["height"];
	auto widthNode = newRoot["width"];
	auto imagePathNode = newRoot["imagePath"];
	Rt = Eigen::Matrix4f::Identity();
	K = Eigen::Matrix4f::Identity();
	if (QtNode.isNull() 
		|| cxNode.isNull()
		|| cyNode.isNull()
		|| fxNode.isNull()
		|| fyNode.isNull()
		|| heightNode.isNull()
		|| widthNode.isNull()
		|| imagePathNode.isNull())
	{
		return  false;
	}
	try
	{
		if (QtNode.isArray() && QtNode.size() == 7)
		{
			double w = QtNode[0].asDouble();
			double x = QtNode[1].asDouble();
			double y = QtNode[2].asDouble();
			double z = QtNode[3].asDouble();
			Eigen::Quaterniond q(w, x, y, z);
			Rt.block(0, 0, 3, 3) = q.matrix();
			Rt(0, 3) = QtNode[4].asDouble();
			Rt(1, 3) = QtNode[5].asDouble();
			Rt(2, 3) = QtNode[6].asDouble();
		}
		K(0, 0) = fxNode.asFloat();
		K(1, 1) = fyNode.asFloat();
		K(0, 2) = cxNode.asFloat();
		K(1, 2) = cyNode.asFloat();
		height = heightNode.asInt();
		width = widthNode.asInt();
		imgPath = imagePathNode.asString();
		return true;
	}
	catch (const std::exception&)
	{
		return false;
	}
	return true;
}

int load_cameras(const std::filesystem::path& resultDir, 
	std::vector<cv::Mat>& imgs,
	std::vector<cv::Mat>& masks,
	std::vector<Eigen::Matrix4f>& Rts,
	std::vector<Eigen::Matrix4f>& Ks)
{
	imgs.reserve(64);
	masks.reserve(64);
	Rts.reserve(64);
	Ks.reserve(64);
	for (auto const& dir_entry : std::filesystem::directory_iterator{ resultDir })
	{
		if (dir_entry.is_regular_file())
		{ 
			const auto& filePath = dir_entry.path();
			if (filePath.has_extension() && filePath.extension().compare(".json")==0)
			{
				Eigen::Matrix4f Rt;
				Eigen::Matrix4f K;
				int height;
				int width;
				std::filesystem::path imgPath;
				LOG_OUT << "try load : " << filePath;
				if (readRsultJson(filePath, Rt, K, height, width, imgPath))
				{
					std::string filename = imgPath.filename().stem().string();
					std::filesystem::path maskPath = imgPath.parent_path() / ("mask_" + filename + ".dat");
					if (std::filesystem::exists(imgPath)&& std::filesystem::exists(maskPath))
					{
						cv::Mat mask = tools::loadMask(maskPath.string());
						cv::Mat img = cv::imread(imgPath.string());
						if (!img.empty() && !mask.empty())
						{
							imgs.emplace_back(img);
							masks.emplace_back(mask);
							Rts.emplace_back(Rt);
							Ks.emplace_back(K);
						}
					}
				};
			}
		} 
	}
	
	return 0;
}

int test_iter_d()
{
	std::vector<cv::Mat>imgs;
	std::vector<cv::Mat>masks;
	std::vector<Eigen::Matrix4d>Rts;
	std::vector<Eigen::Matrix4d>Ks;
	int loadRet = load_cameras("../data/a/result", imgs, masks, Rts, Ks);
	Eigen::MatrixXd V; 
	Eigen::MatrixXi F;
	bool readRet = igl::readOBJ("../InsMesh/out.obj",V,F);
	LOG_OUT << F.cols();
	LOG_OUT << F.rows();


	for (int imgI = 0; imgI < imgs.size(); imgI++)
	{
		const Eigen::Matrix4f& Rt = Rts[imgI];
		const Eigen::Matrix4f& K = Ks[imgI];
		Eigen::Vector3f ray_origin;
		ray_origin[0] = Rt(0, 3);
		ray_origin[1] = Rt(1, 3);
		ray_origin[2] = Rt(2, 3);
		for (int vI = 0; vI < V.rows(); vI++)
		{
			Eigen::Vector3f ray_direction = V.row(vI);
			ray_direction -= ray_origin;
			float dist = ray_direction.norm();
			ray_direction /= dist;
			std::vector<igl::Hit> hits;
			igl::ray_mesh_intersect(ray_origin, ray_direction, V, F, hits);

			for (int vI = 0; vI < V.rows(); vI++)
			{

			}
		}
	}



	return 0;
}