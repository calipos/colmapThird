#include <fstream>
#include <filesystem>
#include <vector>
#include <unordered_map>
#include <string>
#include <functional>
#include <sstream>
#include <list>
#include "Eigen/Core"
#include "Eigen/Geometry"
#include "json/json.h"
#include "opencv2/opencv.hpp"
#include "opencvTools.h"
#include "log.h"
namespace surf
{
	bool readObj(const std::filesystem::path&objPath, Eigen::Matrix4Xf& vertex, Eigen::Matrix3Xi& faces)
	{
		if (!std::filesystem::exists(objPath))
		{
			LOG_ERR_OUT << "not found : " << objPath;
			return false;
		}
		std::fstream fin(objPath, std::ios::in);
		std::string aline;
		std::list<Eigen::Vector3f>v;
		std::list<Eigen::Vector3i>f;
		while (std::getline(fin,aline))
		{
			if (aline.length()>2 && aline[0] == 'v'&& aline[1] == ' ')
			{
				std::stringstream ss(aline);
				char c;
				float x, y, z;
				ss >> c >> x >> y >> z;
				v.emplace_back(x,y,z);
			}
			if (aline.length() > 2 && aline[0] == 'f' && aline[1] == ' ')
			{
				std::stringstream ss(aline);
				char c;
				int x, y, z;
				ss >> c >> x >> y >> z;
				f.emplace_back(x-1, y - 1, z - 1);
			}
		}
		vertex = Eigen::Matrix4Xf(4, v.size());
		faces = Eigen::Matrix3Xi(3, f.size());
		int i = 0;
		for (const auto&d:v)
		{
			vertex(0, i) = d.x();
			vertex(1, i) = d.y();
			vertex(2, i) = d.z();
			vertex(3, i) = 1;
			i += 1;
		}
		i = 0;
		for (const auto& d : f)
		{
			faces(0, i) = d.x();
			faces(1, i) = d.y();
			faces(2, i) = d.z();
			i += 1;
		}
		return true;
	}
	struct Camera
	{
		Camera() {}
		Camera(const double& fx_, const double& fy_, const double& cx_, const double& cy_, const int& height_, const int& width_) 
		{
			fx = fx_;
			fy = fy_;
			cx = cx_;
			cy = cy_;
			height = height_;
			width = width_;
			intr = Eigen::Matrix4f::Identity();
			intr(0, 0) = fx;
			intr(1, 1) = fy;
			intr(0, 2) = cx;
			intr(1, 2) = cy;
		}
		static size_t figureCameraHash(const Camera& c)
		{
			std::stringstream ss;
			ss << static_cast<int>(c.fx * 10000)
				<< static_cast<int>(c.fy * 10000)
				<< static_cast<int>(c.cx * 10000)
				<< static_cast<int>(c.cy * 10000)
				<< static_cast<int>(c.height)
				<< static_cast<int>(c.height);
			std::hash<std::string> hash_fn;
			return hash_fn(ss.str());
		}
		double fx, fy, cx, cy;
		int height, width;
		Eigen::Matrix4f intr;
	};
	struct SurfData
	{
		SurfData() {}
		SurfData(const std::filesystem::path&dataDir_, const std::filesystem::path& objPath_) 
		{
			dataDir = dataDir_;
			objPath = objPath_;
			bool loadObj = surf::readObj("D:/repo/colmapThird/data/a/result/dense.obj", this->vertex, this->faces);
			for (auto const& dir_entry : std::filesystem::recursive_directory_iterator{ dataDir_ })
			{
				const auto& thisFilename = dir_entry.path();
				if (thisFilename.has_extension())
				{
					const auto& ext = thisFilename.extension().string();
					if (ext.compare(".json") == 0)
					{
						std::stringstream ss;
						std::string aline;
						std::fstream fin(thisFilename, std::ios::in);
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
							LOG_ERR_OUT << "parse json fail.";
							return;
						}
						auto newMemberNames = newRoot.getMemberNames();
						if (std::find(newMemberNames.begin(), newMemberNames.end(), "imagePath") == newMemberNames.end())
						{
							continue;
						}
						if (std::find(newMemberNames.begin(), newMemberNames.end(), "fx") == newMemberNames.end())
						{
							continue;
						}
						if (std::find(newMemberNames.begin(), newMemberNames.end(), "fy") == newMemberNames.end())
						{
							continue;
						}
						if (std::find(newMemberNames.begin(), newMemberNames.end(), "cx") == newMemberNames.end())
						{
							continue;
						}
						if (std::find(newMemberNames.begin(), newMemberNames.end(), "cy") == newMemberNames.end())
						{
							continue;
						}
						if (std::find(newMemberNames.begin(), newMemberNames.end(), "height") == newMemberNames.end())
						{
							continue;
						}
						if (std::find(newMemberNames.begin(), newMemberNames.end(), "width") == newMemberNames.end())
						{
							continue;
						}
						if (std::find(newMemberNames.begin(), newMemberNames.end(), "Qt") == newMemberNames.end())
						{
							continue;
						}
						std::filesystem::path imgPath;
						double fx = 0;
						double fy = 0;
						double cx = 0;
						double cy = 0;
						int height = 0;
						int width = 0;
						Eigen::Matrix4f Rt = Eigen::Matrix4f::Identity();
						try
						{
							imgPath = std::filesystem::path(newRoot["imagePath"].asString()); 
							imgPath = std::filesystem::canonical(imgPath);
							fx = newRoot["fx"].asDouble();
							fy = newRoot["fy"].asDouble();
							cx = newRoot["cx"].asDouble();
							cy = newRoot["cy"].asDouble();
							height = newRoot["height"].asInt();
							width = newRoot["width"].asInt();
							if (7!= newRoot["Qt"].size())
							{
								throw std::exception();
							}
							auto QtArray = newRoot["Qt"];
							float w = QtArray[0].asFloat();
							float x = QtArray[1].asFloat();
							float y = QtArray[2].asFloat();
							float z = QtArray[3].asFloat();
							Eigen::Quaternionf q(w, x, y, z);
							Rt.block(0, 0, 3, 3) = q.matrix();
							Rt(0, 3) = QtArray[4].asFloat();
							Rt(1, 3) = QtArray[5].asFloat();
							Rt(2, 3) = QtArray[6].asFloat();
						}
						catch (...)
						{
							LOG_ERR_OUT << "parse json fail : " << thisFilename;
							continue;
						}
						Camera thisCamera(fx, fy, cx, cy, height, width);
						size_t thisCameraSn = Camera::figureCameraHash(thisCamera);
						if (cameras.count(thisCameraSn)==0)
						{
							cameras[thisCameraSn] = thisCamera;
						}
						if (!std::filesystem::exists(imgPath))
						{
							LOG_ERR_OUT << "not found : " << imgPath;
							continue;
						}
						auto shortName = imgPath.filename().stem().string();
						std::filesystem::path maskPath = imgPath.parent_path() / ("mask_"+ shortName+".dat");
						if (!std::filesystem::exists(maskPath))
						{
							LOG_ERR_OUT << "not found : " << maskPath;
							continue;
						}
						cv::Mat img = cv::imread(imgPath.string());
						cv::Mat mask = tools::loadMask(maskPath.string());
						if (img.rows != mask.rows || img.cols != mask.cols )
						{
							LOG_ERR_OUT << "img.rows != mask.rows || img.cols != mask.cols";
							continue;
						}

						cv::Mat rayDistance = cv::Mat::ones(mask.size(), CV_32FC1) * -1;
						Eigen::Matrix4f KRt = thisCamera.intr* Rt;
						std::vector<cv::Vec2i>vertexInImg(this->vertex.cols());
						std::vector<float>vertexDist(this->vertex.cols());
						struct FaceTriangle
						{
							FaceTriangle() {}
							FaceTriangle(const cv::Vec2i& xy0, const cv::Vec2i& xy1, const cv::Vec2i& xy2)
							{
								y1_y2 = xy1[1] - xy2[1];
								x2_x1 = xy2[0] - xy1[0];
								x0_x2 = xy0[0] - xy2[0];
								y0_y2 = xy0[1] - xy2[1];
								y2_y0 = xy2[1] - xy0[1]; 
								denominatorInv = 1/((y1_y2 * x0_x2 + x2_x1 * y0_y2) + 1e-7);
								startX = (std::min)((std::min)(xy0[0], xy1[0]), xy2[0]);
								startY = (std::min)((std::min)(xy0[1], xy1[1]), xy2[1]);
								endX = (std::max)((std::max)(xy0[0], xy1[0]), xy2[0]);
								endY = (std::max)((std::max)(xy0[1], xy1[1]), xy2[1]);
							}
							int startX;
							int startY;
							int endX;
							int endY;
							float y1_y2;
							float x2_x1;
							float x0_x2;
							float y0_y2;
							float y2_y0;
							float denominatorInv;
						};
						{
							Eigen::Matrix4Xf vertexInView = KRt* this->vertex;
#pragma omp parallel for
							for (int i = 0; i < this->vertex.cols(); i++)
							{
								vertexInImg[i][0] = vertexInView(0, i) / vertexInView(2, i) + 0.5;
								vertexInImg[i][1] = vertexInView(1, i) / vertexInView(2, i) + 0.5;
								vertexDist[i] = sqrt(vertexInView(0, i) * vertexInView(0, i) + vertexInView(1, i) * vertexInView(1, i) + vertexInView(2, i) * vertexInView(2, i));
							}
						}
						LOG_OUT;
						this->vertex, this->faces;


					}
				}
			}

		}
		std::filesystem::path dataDir;
		std::filesystem::path objPath;
		Eigen::Matrix4Xf vertex;
		Eigen::Matrix3Xi faces;
		std::unordered_map<size_t, Camera>cameras;
	};
}

int test_surf()
{
	//Eigen::Matrix4Xf vertex; 
	//Eigen::Matrix3Xi faces;
	//bool loadObj = surf::readObj("D:/repo/colmapThird/data/a/result/dense.obj", vertex, faces);

	surf::SurfData asd("D:/repo/colmapThird/data/a/result", "D:/repo/colmapThird/data/a/result/dense.obj");


	return 0;
}