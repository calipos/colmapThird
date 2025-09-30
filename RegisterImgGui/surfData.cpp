#include <numeric>
#include <fstream>
#include <filesystem>
#include <vector>
#include <unordered_map>
#include <unordered_set>
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
	bool transform(const Eigen::MatrixXf& dataIn, std::vector<char>&dataOut)
	{
		int r = dataIn.rows();
		int c = dataIn.cols();
		dataOut.resize(2 * sizeof(int) + r * c * sizeof(float));
		*(int*)&dataOut[0] = r;
		*(int*)&dataOut[sizeof(int)] = c;
		int size = r * c;
		float* data_ = (float*)&dataOut[2 * sizeof(int)];
		for (int i = 0; i < size; i++)
		{
			data_[i] = dataIn(i / c, i % c);
		}
		return true;
	}
	bool transform(const Eigen::MatrixXi& dataIn, std::vector<char>&dataOut)
	{
		int r = dataIn.rows();
		int c = dataIn.cols();
		dataOut.resize(2 * sizeof(int) + r * c * sizeof(int));
		*(int*)&dataOut[0] = r;
		*(int*)&dataOut[sizeof(int)] = c;
		int size = r * c;
		int* data_ = (int*)&dataOut[2 * sizeof(int)];
		for (int i = 0; i < size; i++)
		{
			data_[i] = dataIn(i / c, i % c);
		}
		return true;
	}
	int transform(const char* dataIn, Eigen::MatrixXf& dataOut)
	{
		const int& r = *(int*)&dataIn[0];
		const int& c = *(int*)&dataIn[sizeof(int)];
		dataOut.resize(r, c);
		const float* data_ = (float*)&dataIn[2 * sizeof(int)];
		int size = r * c;
		for (int i = 0; i < size; i++)
		{
			dataOut(i / c, i % c) = data_[i];
		}
		return 2 * sizeof(int)+ size * sizeof(float);
	}
	int transform(const char* dataIn, Eigen::MatrixXi& dataOut)
	{
		const int& r = *(int*)&dataIn[0];
		const int& c = *(int*)&dataIn[sizeof(int)];
		dataOut.resize(r, c);
		const int* data_ = (int*)&dataIn[2 * sizeof(int)];
		int size = r * c;
		for (int i = 0; i < size; i++)
		{
			dataOut(i / c, i % c) = data_[i];
		}
		return 2 * sizeof(int) + size * sizeof(float);
	}
	bool transform(const std::string&line,std::vector<char>&data)
	{
		data.resize(sizeof(int)+ line.length());
		int& length = *(int*)&data[0];
		length = line.length();
		for (int i = 0; i < length; i++)
		{
			data[sizeof(int)+i]= line[i];
		}
		return true;
	}
	int transform(const char* data,std::string& line)
	{
		const int& strLength = *(int*)&data[0];
		line.resize(strLength);
		for (int i = 0; i < strLength; i++)
		{
			 line[i]= data[sizeof(int) + i];
		}
		return strLength+sizeof(int);
	}
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
			int xmin = (std::min)((std::min)(p0[0], p1[0]), p2[0]);
			int xmax = (std::max)((std::max)(p0[0], p1[0]), p2[0]);
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
		return ret;
	}
	struct GridConfig
	{
		float amplitudeHalf;
		float gridUnit;
		int gridLevelCnt;		
		std::vector<char>serialization()const
		{
			std::vector<char> ret(sizeof(float) * 2 + sizeof(int));
			*(float*)&ret[0] = amplitudeHalf;
			*(float*)&ret[sizeof(float)] = gridUnit;
			*(int*)&ret[sizeof(float)*2] = gridLevelCnt;
			return ret;
		}
		int deserialization(const char*data)
		{
			amplitudeHalf = *(float*)&data[0];
			gridUnit = *(float*)&data[sizeof(float)];
			gridLevelCnt = *(int*)&data[sizeof(float) * 2];
			return sizeof(float) * 2 + sizeof(int);
		}
	};
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
		std::vector<char>serialization()const
		{
			std::vector<char>ret(6*sizeof(float));
			float* data = (float*)&ret[0];
			data[0] = fx;
			data[1] = fy;
			data[2] = cx;
			data[3] = cy;
			data[4] = height;
			data[5] = width;
			return ret;
		}
		int deserialization(const char*data)
		{
			const float* floatData = (float*)&data[0];
			fx = floatData[0];
			fy = floatData[1];
			cx = floatData[2];
			cy = floatData[3] ;
			height = floatData[4];
			width = floatData[5] ;
			intr = Eigen::Matrix4f::Identity();
			intr(0, 0) = fx;
			intr(1, 1) = fy;
			intr(0, 2) = cx;
			intr(1, 2) = cy;
			return 6 * sizeof(float);
		}
		static std::vector<char>serialization(const std::unordered_map<size_t, Camera>& cameras)
		{
			std::vector<char>ret(sizeof(int));;
			*(int*)&ret[0] = cameras.size();
			for (const auto&d : cameras)
			{
				std::vector<char>ret0(sizeof(size_t));
				*(size_t*)&ret0[0] = d.first;
				std::vector<char>ret1 = d.second.serialization();
				ret.insert(ret.end(), ret0.begin(), ret0.end());
				ret.insert(ret.end(), ret1.begin(), ret1.end());
			}
			return ret;
		}
		static int deserialization(const char* data, std::unordered_map<size_t, Camera>& dataOut)
		{
			dataOut.clear();
			int elemCnt = *(int*)data;
			int pos = sizeof(int);
			for (int i = 0; i < elemCnt; i++)
			{
				size_t cameraId = *(size_t*)&data[pos]; pos += sizeof(size_t);
				dataOut[cameraId] = Camera();
				int deserialLength = dataOut[cameraId].deserialization(&data[pos]); pos += deserialLength;
			}
			return pos;
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
		cv::Mat getImgDirs()
		{
			if (imgDirs.empty())
			{
				float fxInv = 1. / fx;
				float fyInv = 1. / fy;
				imgDirs = cv::Mat::zeros(height, width, CV_32FC3);
				for (int r = 0; r < height; r++)
				{
					for (int c = 0; c < width; c++)
					{
						Eigen::Vector3f dir((c + 0.5 - cx) * fxInv, (r + 0.5 - cy) * fyInv,1);
						dir.normalize();
						imgDirs.at<cv::Vec3f>(r, c)[0] = dir.x();
						imgDirs.at<cv::Vec3f>(r, c)[1] = dir.y();
						imgDirs.at<cv::Vec3f>(r, c)[2] = dir.z();
					}
				}
			}
			return imgDirs;
		}
		double fx, fy, cx, cy;
		int height, width;
		Eigen::Matrix4f intr;
		cv::Mat imgDirs;
	};
	struct View
	{
		Eigen::Matrix4f Rt;
		std::filesystem::path imgPath;
		std::filesystem::path maskPath;
		std::unordered_map<std::uint32_t, std::unordered_set<std::uint32_t>>pixelGridBelong;
		std::vector<char>serialization()const
		{
			std::vector<char>Rtdata(16*sizeof(float));
			float* Rtdata_ = (float*)&Rtdata[0];
			for (int i = 0; i < 16; i++)
			{
				Rtdata_[i] = Rt(i / 4, i % 4);
			}
			std::vector<char>imgPathData, maskPathData;
			transform(imgPath.string(), imgPathData);
			transform(maskPath.string(), maskPathData);
			std::list<std::uint32_t>pixelGridBelongData;//total:[key:size:element]
			pixelGridBelongData.emplace_back(pixelGridBelong.size());
			for (const auto&d: pixelGridBelong)
			{
				pixelGridBelongData.emplace_back(d.first);
				pixelGridBelongData.emplace_back(d.second.size());
				for (const auto& d2 : d.second)
				{
					pixelGridBelongData.emplace_back(d2);
				}
			}
			std::vector<char>pixelGridBelongData_(pixelGridBelongData.size()*sizeof(std::uint32_t));
			std::uint32_t* data = (std::uint32_t*)&pixelGridBelongData_[0];
			for (const auto&d: pixelGridBelongData)
			{
				*data = d;
				data++;
			}
			Rtdata.insert(Rtdata.end(), imgPathData.begin(), imgPathData.end());
			Rtdata.insert(Rtdata.end(), maskPathData.begin(), maskPathData.end());
			Rtdata.insert(Rtdata.end(), pixelGridBelongData_.begin(), pixelGridBelongData_.end());
			return Rtdata;
		}
		int deserialization(const char*data)
		{
			int pos = 0;
			const float* Rtdata_ = (float*)data;
			for (int i = 0; i < 16; i++)
			{
				Rt(i / 4, i % 4) = Rtdata_[i];
				pos += sizeof(float);
			}
			std::string line;
			pos += transform(&data[pos], line);
			imgPath = std::filesystem::path(line);
			pos += transform(&data[pos], line);
			maskPath = std::filesystem::path(line);
			const std::uint32_t* intData = (const std::uint32_t*)&data[pos];
			std::uint32_t i = 0;
			std::uint32_t mapSize = intData[i]; i++;
			pixelGridBelong.clear();
			for (int pixel = 0; pixel < mapSize; pixel++)
			{
				std::uint32_t key_ = intData[i]; i++;
				std::uint32_t memberSize_ = intData[i]; i++;
				for (std::uint32_t j = 0; j < memberSize_; j++)
				{
					pixelGridBelong[key_].insert(intData[i]); i++;
				}
			}
			return pos+sizeof(std::uint32_t)*i;
		}
		static std::vector<char>serialization(const std::unordered_map<size_t, View>& views)
		{
			std::vector<char>ret(sizeof(int));;
			*(int*)&ret[0] = views.size();
			for (const auto&d: views)
			{
				std::vector<char>ret0(sizeof(size_t));
				*(size_t*)&ret0[0] = d.first;
				std::vector<char>ret1 = d.second.serialization();
				ret.insert(ret.end(), ret0.begin(), ret0.end());
				ret.insert(ret.end(), ret1.begin(), ret1.end());
			}


			//std::unordered_map<size_t, View>tt;
			//deserialization(&ret[0], tt);


			return ret;
		}
		static int deserialization(const char*data, std::unordered_map<size_t, View>&dataOut)
		{
			dataOut.clear();
			int elemCnt = *(int*)data;
			int pos = sizeof(int);
			for (int i = 0; i < elemCnt; i++)
			{
				size_t viewId = *(size_t*)&data[pos]; pos += sizeof(size_t);
				dataOut[viewId] = View();
				int deserialLength = dataOut[viewId].deserialization(&data[pos]); pos += deserialLength;				
			}
			return 0;
		}
	};
	struct SurfData
	{
#define XyzToGridXYZ(x,y,z,xStart,yStart,zStart,gridUnitInv,gridX,gridY,gridZ) \
		int gridX = (x-xStart)*gridUnitInv;  \
		int gridY = (y-yStart)*gridUnitInv;  \
		int gridZ = (z-zStart)*gridUnitInv;  
#define GridXyzToPosEncode(gridX,gridY,gridZ,resolutionX,resolutionXY,posEncode) \
		std::uint32_t posEncode = gridX+gridY*resolutionX+gridZ*resolutionXY;
#define ImgXyToPosEncode(x,y,imgWidth)  (imgWidth*y+x)

		SurfData() {}
		SurfData(const std::filesystem::path&dataDir_, const std::filesystem::path& objPath_, const GridConfig& gridConfig_)
		{
			dataDir = dataDir_;
			objPath = objPath_;
			gridConfig = gridConfig_;
			this->gridUnitInv = 1. / gridConfig.gridUnit;
			this->vertex.resize(0, 0);
			{
				bool loadObj = surf::readObj(objPath, this->vertex, this->faces, this->vertex_normal, this->face_normal);
				if (!loadObj)
				{
					LOG_ERR_OUT << "read obj fail : " << objPath;
				}
				float objMinX = this->vertex.row(0).minCoeff();
				float objMinY = this->vertex.row(1).minCoeff();
				float objMinZ = this->vertex.row(2).minCoeff();
				float objMaxX = this->vertex.row(0).maxCoeff();
				float objMaxY = this->vertex.row(1).maxCoeff();
				float objMaxZ = this->vertex.row(2).maxCoeff();
				this->targetMinBorderX = objMinX - 0.06 * (objMaxX - objMinX);
				this->targetMinBorderY = objMinY - 0.06 * (objMaxY - objMinY);
				this->targetMinBorderZ = objMinZ - 0.06 * (objMaxZ - objMinZ);
				this->targetMaxBorderX = objMaxX + 0.06 * (objMaxX - objMinX);
				this->targetMaxBorderY = objMaxY + 0.06 * (objMaxY - objMinY);
				this->targetMaxBorderZ = objMaxZ + 0.06 * (objMaxZ - objMinZ);

				this->resolutionX = (this->targetMaxBorderX - this->targetMinBorderX) / gridConfig.gridUnit + 1;
				this->resolutionY = (this->targetMaxBorderY - this->targetMinBorderY) / gridConfig.gridUnit + 1;
				this->resolutionZ = (this->targetMaxBorderZ - this->targetMinBorderZ) / gridConfig.gridUnit + 1;
				if (!isValidResolutions(this->resolutionX, this->resolutionY, this->resolutionZ))
				{
					LOG_ERR_OUT << "grid unit too small!!!";
					exit(-1);
				}
				this->resolutionXY = this->resolutionX * this->resolutionY;

			}
			for (auto const& dir_entry : std::filesystem::recursive_directory_iterator{ dataDir_ })
			{
				const auto& thisFilename = dir_entry.path();
				if (thisFilename.has_extension())
				{
					const auto& ext = thisFilename.extension().string();
					if (ext.compare(".json") == 0)
					{
						LOG_OUT << thisFilename;
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
						try
						{


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
						}
						catch (...)
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
							if (7 != newRoot["Qt"].size())
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
						if (this->cameras.count(thisCameraSn) == 0)
						{
							this->cameras[thisCameraSn] = thisCamera;
						}
						if (!std::filesystem::exists(imgPath))
						{
							LOG_ERR_OUT << "not found : " << imgPath;
							continue;
						}
						auto shortName = imgPath.filename().stem().string();
						std::filesystem::path maskPath = imgPath.parent_path() / ("mask_" + shortName + ".dat");
						if (!std::filesystem::exists(maskPath))
						{
							LOG_ERR_OUT << "not found : " << maskPath;
							continue;
						}
						cv::Mat img = cv::imread(imgPath.string());
						cv::Mat mask = tools::loadMask(maskPath.string());
						if (img.rows != mask.rows || img.cols != mask.cols)
						{
							LOG_ERR_OUT << "img.rows != mask.rows || img.cols != mask.cols";
							continue;
						}
						View thisView = { Rt ,imgPath ,maskPath };
						size_t viewIdx = this->views.size();
						this->views[viewIdx]=thisView;
						cv::Mat rayDistance = cv::Mat::ones(mask.size(), CV_32FC1) * -1;
						const Eigen::Matrix3f R = Rt.block(0, 0, 3, 3);
						const Eigen::Matrix4f KRt = thisCamera.intr * Rt;
						std::vector<std::int8_t> visualFace(this->faces.cols(), 0);
						Eigen::Vector3f cameraT(Rt(0, 3), Rt(1, 3), Rt(2, 3));
						{
							std::vector<float>distFromT(this->faces.cols());
#pragma omp parallel for
							for (int i = 0; i < this->faces.cols(); i++)
							{
								const int& v = faces(0, i);//choose the first vertex idx
								const float a = Rt(0, 3) - this->vertex(0, v);
								const float b = Rt(1, 3) - this->vertex(1, v);
								const float c = Rt(2, 3) - this->vertex(2, v);
								float dirSign = a * this->face_normal(0, i) + b * this->face_normal(1, i) + c * this->face_normal(2, i);
								if (dirSign > 0)
								{
									visualFace[i] = 1;
								}
								else if (dirSign < 0)
								{
									visualFace[i] = -1;
								}
								distFromT[i] = a * a + b * b + c * c;
							}
							int neareatFaceIdx = std::min_element(distFromT.begin(), distFromT.end()) - distFromT.begin();
							std::int8_t nearestFaceNormalSigned = visualFace[neareatFaceIdx];
#pragma omp parallel for
							for (int i = 0; i < visualFace.size(); i++)
							{
								visualFace[i] *= nearestFaceNormalSigned;
							}
						}
						std::vector<cv::Vec2i>vertexInImg(this->vertex.cols());
						std::vector<float>vertexDist(this->vertex.cols());
						{
							Eigen::Matrix4Xf vertexInView = Rt * this->vertex;
							std::vector<bool> vertexInViewInCanvas(this->vertex.cols(), false);
							float fxInv = 1. / this->cameras[thisCameraSn].fx;
							float fyInv = 1. / this->cameras[thisCameraSn].fy;
#pragma omp parallel for
							for (int i = 0; i < this->vertex.cols(); i++)
							{
								vertexDist[i] = sqrt(vertexInView(0, i) * vertexInView(0, i) + vertexInView(1, i) * vertexInView(1, i) + vertexInView(2, i) * vertexInView(2, i));
								vertexInImg[i][0] = vertexInView(0, i) / vertexInView(2, i) * cameras[thisCameraSn].fx + cameras[thisCameraSn].cx + 0.5;
								vertexInImg[i][1] = vertexInView(1, i) / vertexInView(2, i) * cameras[thisCameraSn].fy + cameras[thisCameraSn].cy + 0.5;
								if (vertexInImg[i][0] >= 0 && vertexInImg[i][1] >= 0 && vertexInImg[i][0] < width && vertexInImg[i][1] < height)
								{
									vertexInViewInCanvas[i] = true;
								}
							}
							for (int i = 0; i < this->faces.cols(); i++)
							{
								if (visualFace[i] > 0)
								{
									const auto& a = this->faces(0, i);
									const auto& b = this->faces(1, i);
									const auto& c = this->faces(2, i);
									if (vertexInViewInCanvas[a] && vertexInViewInCanvas[b] && vertexInViewInCanvas[c])
									{
										const cv::Vec2i& uv0 = vertexInImg[a];
										const cv::Vec2i& uv1 = vertexInImg[b];
										const cv::Vec2i& uv2 = vertexInImg[c];
										std::list<cv::Vec2i> trianglePixel = triangle(uv0, uv1, uv2);
										for (const auto& d : trianglePixel)
										{
											float& dist = rayDistance.ptr<float>(d[1])[d[0]];
											if (dist<0 || dist>vertexDist[a])
											{
												dist = vertexDist[a];
											}
										}
									}
								}
							}
						}
						if (false)
						{//show rayDistant
							Eigen::MatrixXf pts = rayDistantMapToPts(rayDistance, this->cameras[thisCameraSn], Rt);
							tools::saveColMajorPts3d("../surf/a.ply", pts);
						}
						this->registerRayDistanceMap(rayDistance, this->cameras[thisCameraSn], this->views[viewIdx]);
					}
				}
			}
			std::vector<char>d = this->serialization();
			std::fstream fout("../surf/d.dat",std::ios::out|std::ios::binary);
			int dCnt = d.size();
			fout.write((char*)&dCnt, sizeof(int));
			fout.write(&d[0], dCnt* sizeof(char));
			fout.close();
		}
		static bool isValidResolutions(const int& resolutionX, const int& resolutionY, const int& resolutionZ)
		{
			if (resolutionX <= 0 || resolutionY <= 0 || resolutionZ <= 0)
			{
				return false;
			}
			int xy = resolutionX * resolutionY;
			int xz = resolutionX * resolutionZ;
			int zy = resolutionZ * resolutionY;
			if ((xy / resolutionX != resolutionY) || (xz / resolutionX != resolutionZ) || (zy / resolutionY != resolutionZ))
			{
				return false;
			}
			int xyz = xy * resolutionZ;
			if (xyz/ xy != resolutionZ)
			{
				return false;
			}
			return true;
		}
		bool registerRayDistanceMap(const cv::Mat& rayDistantMap, Camera& camera, View& view)
		{//pixelGridBelong
			cv::Mat imgDirs = camera.getImgDirs();
			const Eigen::Matrix4f Rtinv = view.Rt.inverse();
			const Eigen::Matrix3f Rinv = Rtinv.block(0, 0, 3, 3);
			const Eigen::Vector3f tinv(Rtinv(0, 3), Rtinv(1, 3), Rtinv(2, 3));
			int amplitudeCnt = 2 * this->gridConfig.amplitudeHalf / this->gridConfig.gridUnit + 1;
			Eigen::MatrixXf pixelViewPts(4, amplitudeCnt);
			{				
				for (int r = 0; r < camera.height; r++)
				{
					for (int c = 0; c < camera.width; c++)
					{
						const float& rayDist = rayDistantMap.ptr<float>(r)[c];
						if (1e-3 < rayDist)
						{
							std::uint32_t imgPixelPos = ImgXyToPosEncode(c, r, camera.width);
							for (int i = 0; i < amplitudeCnt; i++)
							{
								Eigen::Vector3f p = (rayDist + -this->gridConfig.amplitudeHalf + i * this->gridConfig.gridUnit) * Eigen::Vector3f(imgDirs.at<cv::Vec3f>(r, c)[0], imgDirs.at<cv::Vec3f>(r, c)[1], imgDirs.at<cv::Vec3f>(r, c)[2]);
								if (p[0] < this->targetMinBorderX || p[1] < this->targetMinBorderY || p[2] < this->targetMinBorderZ
									|| p[0] >= this->targetMaxBorderX || p[1] >= this->targetMaxBorderY || p[2] >= this->targetMaxBorderZ)
								{
									continue;
								}
								Eigen::Vector3f p1 = Rinv * p + tinv;
								XyzToGridXYZ(p1[0], p1[0], p1[2], this->targetMinBorderX, this->targetMinBorderY, this->targetMinBorderZ, this->gridUnitInv, gridX, gridY, gridZ);
								GridXyzToPosEncode(gridX, gridY, gridZ, this->resolutionX, this->resolutionXY, posEncode);
								view.pixelGridBelong[imgPixelPos].insert(posEncode);
							}
						}
					}
				}

			}

			return true;
		}
		static Eigen::MatrixXf rayDistantMapToPts(const cv::Mat& rayDistantMap, Camera& camera, const Eigen::Matrix4f& viewRt)
		{
			Eigen::MatrixXf ret;
			cv::Mat imgDirs = camera.getImgDirs();
			{
				std::list<Eigen::Vector3f>listRet;
				for (int r = 0; r < camera.height; r++)
				{
					for (int c = 0; c < camera.width; c++)
					{
						const float& rayDist = rayDistantMap.ptr<float>(r)[c];
						if (1e-3 < rayDist)
						{
							listRet.emplace_back(rayDist * Eigen::Vector3f(imgDirs.at<cv::Vec3f>(r, c)[0], imgDirs.at<cv::Vec3f>(r, c)[1], imgDirs.at<cv::Vec3f>(r, c)[2]));
						}
					}
				}
				ret.resize(4, listRet.size());
				int i = 0;
				for (const auto&d: listRet)
				{
					ret(0, i) = d.x();
					ret(1, i) = d.y();
					ret(2, i) = d.z();
					ret(3, i) = 1;
					i += 1;
				}
			}
			const Eigen::Matrix4f Rtinv = viewRt.inverse();
			return Rtinv* ret;
		}
		std::vector<char>serialization()const
		{
			std::vector<char>ret;
			std::vector<char>imgPathData, maskPathData, vertexData, facesData, vertex_normalData, face_normalData;
			transform(this->dataDir.string(), imgPathData);
			transform(this->objPath.string(), maskPathData);
			transform(this->vertex, vertexData);
			transform(this->faces, facesData);
			transform(this->vertex_normal, vertex_normalData);
			transform(this->face_normal, face_normalData);
			ret.insert(ret.end(), imgPathData.begin(), imgPathData.end());
			ret.insert(ret.end(), maskPathData.begin(), maskPathData.end());
			ret.insert(ret.end(), vertexData.begin(), vertexData.end());
			ret.insert(ret.end(), facesData.begin(), facesData.end());
			ret.insert(ret.end(), vertex_normalData.begin(), vertex_normalData.end());
			ret.insert(ret.end(), face_normalData.begin(), face_normalData.end());
			{
				std::vector<char>floatDat(7*sizeof(float));
				*(float*)&floatDat[0 * sizeof(float)]= targetMinBorderX;
				*(float*)&floatDat[1 * sizeof(float)] = targetMinBorderY;
				*(float*)&floatDat[2 * sizeof(float)] = targetMinBorderZ;
				*(float*)&floatDat[3 * sizeof(float)] = targetMaxBorderX;
				*(float*)&floatDat[4 * sizeof(float)] = targetMaxBorderY;
				*(float*)&floatDat[5 * sizeof(float)]= targetMaxBorderZ;
				*(float*)&floatDat[6 * sizeof(float)] = gridUnitInv;
				std::vector<char>intDat(4 * sizeof(float));
				*(int*)&intDat[0 * sizeof(int)] = resolutionX;
				*(int*)&intDat[1 * sizeof(int)] = resolutionY;
				*(int*)&intDat[2 * sizeof(int)] = resolutionZ;
				*(int*)&intDat[3 * sizeof(int)] = resolutionXY;
				ret.insert(ret.end(), floatDat.begin(), floatDat.end());
				ret.insert(ret.end(), intDat.begin(), intDat.end());
			}
			std::vector<char>configDat = gridConfig.serialization();
			ret.insert(ret.end(), configDat.begin(), configDat.end());
			std::vector<char>camerasData = Camera::serialization(this->cameras);
			ret.insert(ret.end(), camerasData.begin(), camerasData.end());
			std::vector<char>viewData = View::serialization(this->views);
			ret.insert(ret.end(), viewData.begin(), viewData.end());
			return ret;
		}
		bool reload(const std::filesystem::path&path)
		{
			std::fstream fin(path,std::ios::in|std::ios::binary);
			int totalSize = 0;
			fin.read((char*)&totalSize,sizeof(int));
			std::vector<char>dat(totalSize);
			fin.read((char*)&dat[0], totalSize* sizeof(char));
			int pos = 0;
			std::string line;
			pos += transform(&dat[pos], line); this->dataDir = std::filesystem::path(line);
			pos += transform(&dat[pos], line); this->objPath = std::filesystem::path(line);
			pos += transform(&dat[pos], this->vertex);
			pos += transform(&dat[pos], this->faces);
			pos += transform(&dat[pos], this->vertex_normal);
			pos += transform(&dat[pos], this->face_normal);
			{
				targetMinBorderX = *(float*)&dat[pos]; pos += sizeof(float);
				targetMinBorderY = *(float*)&dat[pos]; pos += sizeof(float);
				targetMinBorderZ = *(float*)&dat[pos]; pos += sizeof(float);
				targetMaxBorderX = *(float*)&dat[pos]; pos += sizeof(float);
				targetMaxBorderY = *(float*)&dat[pos]; pos += sizeof(float);
				targetMaxBorderZ = *(float*)&dat[pos]; pos += sizeof(float);
				gridUnitInv = *(float*)&dat[pos]; pos += sizeof(float);
				resolutionX = *(int*)&dat[pos]; pos += sizeof(int);
				resolutionY = *(int*)&dat[pos]; pos += sizeof(int);
				resolutionZ = *(int*)&dat[pos]; pos += sizeof(int);
				resolutionXY = *(int*)&dat[pos]; pos += sizeof(int);
			}
			pos += gridConfig.deserialization(&dat[pos]);
			pos += Camera::deserialization(&dat[pos], this->cameras);
			pos += View::deserialization(&dat[pos], this->views);
			return true;
		}
		std::filesystem::path dataDir;
		std::filesystem::path objPath;
		Eigen::MatrixXf vertex;
		Eigen::MatrixXi faces;
		Eigen::MatrixXf vertex_normal;
		Eigen::MatrixXf face_normal;
		float targetMinBorderX;
		float targetMinBorderY;
		float targetMinBorderZ;
		float targetMaxBorderX;
		float targetMaxBorderY;
		float targetMaxBorderZ;
		float gridUnitInv;
		int resolutionX;
		int resolutionY;
		int resolutionZ;
		int resolutionXY;
		std::unordered_map<size_t, Camera>cameras;
		std::unordered_map < size_t, View>views;
		GridConfig gridConfig;
		
	};
} 
int test_surf()
{
	float totalAmplitude = 0.3;//measured from obj data manually
	float gridUnit = totalAmplitude / 10;// the Amplitude, i wang to separet it into 10 picecs
	surf::GridConfig gridConfig = { totalAmplitude /2,gridUnit ,4};
	surf::SurfData asd("D:/repo/colmapThird/data/a/result", "D:/repo/colmapThird/data/a/result/dense.obj", gridConfig);

	//surf::SurfData asd2;
	//asd2.reload("../surf/d.dat");
	return 0;
}