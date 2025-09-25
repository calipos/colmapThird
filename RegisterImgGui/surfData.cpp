#include <numeric>
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
#include "igl/per_vertex_normals.h"
#include "igl/per_face_normals.h"
#include "log.h"
namespace surf
{
	bool readObj(const std::filesystem::path&objPath, Eigen::MatrixXf& vertex, Eigen::MatrixXi& faces, Eigen::MatrixXf& vertex_normal, Eigen::MatrixXf& face_normal)
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
		vertex = Eigen::MatrixXf( v.size(),3);
		faces = Eigen::MatrixXi( f.size(),3);
		int i = 0;
		for (const auto&d:v)
		{
			vertex(i, 0) = d.x();
			vertex(i, 1) = d.y();
			vertex(i, 2) = d.z();
			i += 1;
		}
		i = 0;
		for (const auto& d : f)
		{
			faces(i, 0) = d.x();
			faces(i, 1) = d.y();
			faces(i, 2) = d.z();
			i += 1;
		}
		igl::per_vertex_normals(vertex, faces, vertex_normal);
		igl::per_face_normals(vertex, faces, face_normal);
//		if (cameraT != nullptr)
//		{
//			std::vector<float>distFromT(f.size());
//#pragma omp parallel for
//			for (int i = 0; i < f.size(); i++)
//			{
//				const int& v = faces(i, 0);//choose the first vertex idx
//				float a = vertex(v, 0) - cameraT->x();
//				float b = vertex(v, 1) - cameraT->y();
//				float c = vertex(v, 2) - cameraT->z();
//				distFromT[i] = a * a + b * b + c * c;
//			}
//			int neareatFaceIdx = std::min_element(distFromT.begin(), distFromT.end()) - distFromT.begin();
//			int neareatVertexIdx = faces(neareatFaceIdx, 0);
//			Eigen::Vector3f vertexNormal(vertex_normal(neareatVertexIdx, 0), vertex_normal(neareatVertexIdx, 1), vertex_normal(neareatVertexIdx, 2));
//			Eigen::Vector3f cameraTDir(vertex(neareatVertexIdx, 0) - cameraT->x(), vertex(neareatVertexIdx, 1) - cameraT->y(), vertex(neareatVertexIdx, 2) - cameraT->z());
//			Eigen::Vector3f faceNormal(face_normal(neareatFaceIdx, 0), face_normal(neareatFaceIdx, 1), face_normal(neareatFaceIdx, 2));
//			cameraTDir.normalize();
//			if (vertexNormal.dot(cameraTDir) < 0)
//			{
//				LOG_OUT << "swtich normals.";
//				vertex_normal *= -1;
//			}
//			if (faceNormal.dot(cameraTDir) < 0)
//			{
//				LOG_OUT << "swtich normals.";
//				face_normal *= -1;
//			}
//		}
		vertex = Eigen::Matrix4Xf(4,v.size()); 
		i = 0;
		for (const auto& d : v)
		{
			vertex(0,i) = d.x();
			vertex(1,i) = d.y();
			vertex(2,i) = d.z();
			vertex(3,i) = 1;
			i += 1;
		}
		faces.transposeInPlace();
		vertex_normal.transposeInPlace();
		face_normal.transposeInPlace();
		return true;
	}
	std::list<cv::Vec2i> triangle(const cv::Vec2i& p0, const cv::Vec2i& p1, const cv::Vec2i& p2) {
		std::list<cv::Vec2i> ret;
		//triangle area = 0
		if (p0[1] == p1[1] && p0[1] == p2[1])
		{
			int xmin = (std::min)((std::min)(p0[0], p1[0]), p1[2]);
			int xmax = (std::max)((std::max)(p0[0], p1[0]), p1[2]);
			for (int i = xmin; i <= xmax; i++)
			{
				ret.emplace_back(i, p0[1]);
			}
			return ret;
		}
		//sort base on Y
		cv::Vec2i t0 = p0;
		cv::Vec2i t1 = p1;
		cv::Vec2i t2 = p2;
		if (t0[1] > t1[1]) std::swap(t0, t1);
		if (t0[1] > t2[1]) std::swap(t0, t2);
		if (t1[1] > t2[1]) std::swap(t1, t2);
		int total_height = t2[1] - t0[1];
		for (int i = 0; i < total_height; i++) {
			//separate
			bool second_half = i > t1[1] - t0[1] || t1[1] == t0[1];
			int segment_height = second_half ? t2[1] - t1[1] : t1[1] - t0[1];
			float alpha = (float)i / total_height;
			float beta = (float)(i - (second_half ? t1[1] - t0[1] : 0)) / segment_height;

			cv::Vec2i A = t0 + (t2 - t0) * alpha;
			cv::Vec2i B = second_half ? t1 + (t2 - t1) * beta : t0 + (t1 - t0) * beta;
			if (A[0] > B[0]) std::swap(A, B);
			for (int j = A[0]; j <= B[0]; j++) {
				ret.emplace_back(j, t0[1] + i);
			}
		}
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
			this->vertex.resize(0, 0);
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
							if (this->vertex.rows()==0)
							{
								bool loadObj = surf::readObj(objPath, this->vertex, this->faces, this->vertex_normal, this->face_normal);
								if (!loadObj)
								{
									LOG_ERR_OUT << "read obj fail : " << objPath;
								}
							}
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
						const Eigen::Matrix3f R = Rt.block(0,0,3,3);
						const Eigen::Matrix4f KRt = thisCamera.intr * Rt;
						std::vector<bool> visualFace(this->faces.cols(),false);
						Eigen::Vector3f cameraT(Rt(0, 3), Rt(1, 3), Rt(2, 3));
						{
							Eigen::MatrixXf faceToCameraT(3, this->faces.cols());
							std::vector<float>distFromT(this->faces.cols());
							
#pragma omp parallel for
							for (int i = 0; i < this->faces.cols(); i++)
							{
								const int& v = faces(i, 0);//choose the first vertex idx
								const float a = this->vertex(0,v) - Rt(0, 3);
								const float b = this->vertex(1,v) - Rt(1, 3);
								const float c = this->vertex(2,v) - Rt(2, 3);
								faceToCameraT(0, i) = a;
								faceToCameraT(1, i) = b;
								faceToCameraT(2, i) = c;
								distFromT[i] = a * a + b * b + c * c;
							}
							int neareatFaceIdx = std::min_element(distFromT.begin(), distFromT.end()) - distFromT.begin();
							Eigen::Vector3f nearestFaceToCameraT(faceToCameraT(0, neareatFaceIdx), faceToCameraT(1, neareatFaceIdx), faceToCameraT(2, neareatFaceIdx));
							Eigen::Vector3f nearestFaceNormal(this->face_normal(0, neareatFaceIdx), this->face_normal(1, neareatFaceIdx), this->face_normal(2, neareatFaceIdx));
							nearestFaceToCameraT.normalize();
							if (nearestFaceToCameraT.dot(nearestFaceNormal) > 0)
							{
								LOG_OUT << "swtich normals.";
								face_normal *= -1;
							}
							this->faces.mul(faceToCameraT;
						}
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
							}
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
		Eigen::MatrixXf vertex;
		Eigen::MatrixXi faces;
		Eigen::MatrixXf vertex_normal;
		Eigen::MatrixXf face_normal;
		std::unordered_map<size_t, Camera>cameras;
	};
} 
int test_surf()
{
 
	surf::SurfData asd("D:/repo/colmapThird/data/a/result", "D:/repo/colmapThird/data/a/result/dense.obj");


	return 0;
}