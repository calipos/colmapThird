#include <functional>
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
#include "log.h"
namespace mrf
{
	bool transform(const Eigen::MatrixXf& dataIn, std::vector<char>& dataOut)
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
	bool transform(const Eigen::MatrixXi& dataIn, std::vector<char>& dataOut)
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
		return 2 * sizeof(int) + size * sizeof(float);
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
	bool transform(const std::string& line, std::vector<char>& data)
	{
		data.resize(sizeof(int) + line.length());
		int& length = *(int*)&data[0];
		length = line.length();
		for (int i = 0; i < length; i++)
		{
			data[sizeof(int) + i] = line[i];
		}
		return true;
	}
	int transform(const char* data, std::string& line)
	{
		const int& strLength = *(int*)&data[0];
		line.resize(strLength);
		for (int i = 0; i < strLength; i++)
		{
			line[i] = data[sizeof(int) + i];
		}
		return strLength + sizeof(int);
	}

	struct GridConfig
	{
		int rayPixelCnt;
		float gridResolution;

		std::vector<char>serialization()const
		{
			std::vector<char> ret(sizeof(float) + sizeof(int));
			*(int*)&ret[0] = rayPixelCnt;
			*(float*)&ret[sizeof(int)] = gridResolution;
			return ret;
		}
		int deserialization(const char* data)
		{
			rayPixelCnt = *(int*)&data[0];
			gridResolution = *(float*)&data[sizeof(int)];
			return sizeof(float) + sizeof(int);
		}
	};
	bool isValidResolutions(const int& resolutionX, const int& resolutionY, const int& resolutionZ)
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
		if (xyz / xy != resolutionZ)
		{
			return false;
		}
		return true;
	}
	bool readObj(const std::filesystem::path& objPath, Eigen::MatrixXf& vertex, Eigen::MatrixXi& faces)
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
		while (std::getline(fin, aline))
		{
			if (aline.length() > 2 && aline[0] == 'v' && aline[1] == ' ')
			{
				std::stringstream ss(aline);
				char c;
				float x, y, z;
				ss >> c >> x >> y >> z;
				v.emplace_back(x, y, z);
			}
			if (aline.length() > 2 && aline[0] == 'f' && aline[1] == ' ')
			{
				std::stringstream ss(aline);
				char c;
				int x, y, z;
				ss >> c >> x >> y >> z;
				f.emplace_back(x - 1, y - 1, z - 1);
			}
		}
		vertex = Eigen::MatrixXf(v.size(), 3);
		faces = Eigen::MatrixXi(f.size(), 3);
		int i = 0;
		for (const auto& d : v)
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
		vertex = Eigen::Matrix4Xf(4, v.size());
		i = 0;
		for (const auto& d : v)
		{
			vertex(0, i) = d.x();
			vertex(1, i) = d.y();
			vertex(2, i) = d.z();
			vertex(3, i) = 1;
			i += 1;
		}
		faces.transposeInPlace();
		return true;
	}
	std::pair<float, float> calVarStdev(std::vector<float> vecNums) {
		std::pair<float, float> res;
		float sumNum = accumulate(vecNums.begin(), vecNums.end(), 0.0);
		float mean = sumNum / vecNums.size(); // 计算均值
		float accum = 0.0;
		for_each(vecNums.begin(), vecNums.end(), [&](const float d) {
			accum += (d - mean) * (d - mean);
			});
		float variance = accum / vecNums.size(); // 计算方差
		res.first = mean;
		res.second = variance;
		return res;
	}
	bool generThinMesh(const cv::Mat& mask, const cv::Mat& ptsMat, std::vector<cv::Point3f>& pts, std::vector<cv::Point3i>& faces, const cv::Point3f& backDir = cv::Point3f(0, 0, 1),const float&thin=0.1)
	{
		if (mask.size() !=ptsMat.size())
		{
			LOG_ERR_OUT << "mask.size() !=ptsMat.size()";
			return false;
		}
		if (mask.type() != CV_8UC1&& mask.type() != CV_32FC3)
		{ 
			LOG_ERR_OUT << "mask.type() != CV_8UC1&& mask.type() != CV_32FC3";
			return false;
		}
		int height = mask.rows;
		int width = mask.cols;
		int height_1 = height - 1;
		int width_1 = width - 1;
		int fontPtsCnt = cv::countNonZero(mask);
		pts.clear();
		pts.reserve(fontPtsCnt*2);
		std::list<cv::Point3i>faces_;
		std::list<cv::Point2i>borders;
		cv::Mat indexMat = cv::Mat::ones(ptsMat.size(), CV_32SC1)*-1;
		int idx = 0;
		for (int r = 0; r < height_1; r++)
		{
			for (int c = 0; c < width_1; c++)
			{
				int r_1 = r + 1;
				int c_1 = c + 1;
				const uchar& m_0 = mask.ptr<uchar>(r)[c];
				const uchar& m_1 = mask.ptr<uchar>(r)[c + 1];
				const uchar& m_2 = mask.ptr<uchar>(r + 1)[c];
				const uchar& m_3 = mask.ptr<uchar>(r + 1)[c + 1];
				if (3 > m_0 + m_1 + m_2 + m_3)
				{
					continue;
				}
				int& i_0 = indexMat.ptr<int>(r)[c];
				int& i_1 = indexMat.ptr<int>(r)[c + 1];
				int& i_2 = indexMat.ptr<int>(r + 1)[c];
				int& i_3 = indexMat.ptr<int>(r + 1)[c + 1];
				if (m_0)
				{
					if (i_0 < 0)i_0 = idx;
					pts.emplace_back(ptsMat.at<cv::Vec3f>(r, c));
					idx += 1;
				}
				if (m_1)
				{
					if (i_1 < 0)i_1 = idx;
					pts.emplace_back(ptsMat.at<cv::Vec3f>(r, c + 1)); 
					idx += 1;
				}
				if (m_2)
				{
					if (i_2 < 0)i_2 = idx;
					pts.emplace_back(ptsMat.at<cv::Vec3f>(r + 1, c));
					idx += 1;
				}
				if (m_3)
				{
					if (i_3 < 0)i_3 = idx;
					pts.emplace_back(ptsMat.at<cv::Vec3f>(r + 1, c + 1));
					idx += 1;
				}
				if (!m_0 && m_1 & m_2 && m_3)
				{
					faces_.emplace_back(i_1, i_3, i_2);
					borders.emplace_back(i_2, i_1);
					if (c_1 == width_1 || 0 == mask.ptr<uchar>(r)[c + 2] + mask.ptr<uchar>(r + 1)[c + 2])
					{
						borders.emplace_back(i_1, i_3);
					}
					if (r_1 == height_1 || 0 == mask.ptr<uchar>(r + 2)[c] + mask.ptr<uchar>(r + 2)[c + 1])
					{
						borders.emplace_back(i_3, i_2);
					}
					continue;
				}
				if (m_0 && !m_1 & m_2 && m_3)
				{
					faces_.emplace_back(i_0, i_3, i_2);
					borders.emplace_back(i_0, i_3);
					if (c == 0 || 0 == mask.ptr<uchar>(r)[c - 1] + mask.ptr<uchar>(r + 1)[c - 1])
					{
						borders.emplace_back(i_2, i_0);
					}
					if (r_1 == height_1 || 0 == mask.ptr<uchar>(r + 2)[c] + mask.ptr<uchar>(r + 2)[c + 1])
					{
						borders.emplace_back(i_3, i_2);
					}
					continue;
				}
				if (m_0 && m_1 & !m_2 && m_3)
				{
					faces_.emplace_back(i_0, i_1, i_3);
					borders.emplace_back(i_3, i_0);
					if (c_1 == width_1 || 0 == mask.ptr<uchar>(r)[c + 2] + mask.ptr<uchar>(r + 1)[c + 2])
					{
						borders.emplace_back(i_1, i_3);
					}
					if (r == 0 || 0 == mask.ptr<uchar>(r - 1)[c] + mask.ptr<uchar>(r - 1)[c + 1])
					{
						borders.emplace_back(i_0, i_1);
					}
					continue;
				}
				if (m_0 && m_1 & m_2 && !m_3)
				{
					faces_.emplace_back(i_0, i_1, i_2);
					borders.emplace_back(i_1, i_2);
					if (c == 0 || 0 == mask.ptr<uchar>(r)[c - 1] + mask.ptr<uchar>(r + 1)[c - 1])
					{
						borders.emplace_back(i_2, i_0);
					}
					if (r == 0 || 0 == mask.ptr<uchar>(r - 1)[c] + mask.ptr<uchar>(r - 1)[c + 1])
					{
						borders.emplace_back(i_0, i_1);
					}
					continue;
				}
				if (m_0 && m_1 & m_2 && m_3)
				{
					faces_.emplace_back(i_0, i_1, i_2);
					faces_.emplace_back(i_1, i_3, i_2);
					if (c == 0 || 0 == mask.ptr<uchar>(r)[c - 1] + mask.ptr<uchar>(r + 1)[c - 1])
					{
						borders.emplace_back(i_2, i_0);
					}
					if (r == 0 || 0 == mask.ptr<uchar>(r - 1)[c] + mask.ptr<uchar>(r - 1)[c + 1])
					{
						borders.emplace_back(i_0, i_1);
					}
					if (c_1 == width_1 || 0 == mask.ptr<uchar>(r)[c + 2] + mask.ptr<uchar>(r + 1)[c + 2])
					{
						borders.emplace_back(i_1, i_3);
					}
					if (r_1 == height_1 || 0 == mask.ptr<uchar>(r + 2)[c] + mask.ptr<uchar>(r + 2)[c + 1])
					{
						borders.emplace_back(i_3, i_2);
					}
					continue;
				}
			}
		}
		fontPtsCnt = pts.size();
		cv::Point3f backDirUnit = backDir / cv::norm(backDir) * thin;
		for (int i = 0; i < fontPtsCnt; i++)
		{
			pts.emplace_back(pts[i]+ backDirUnit);
		}
		faces.clear();
		faces.reserve(faces_.size() * 2 + 2 * borders.size());
		for (const auto& d : faces_)
		{
			faces.emplace_back(d);
			faces.emplace_back(d.x + fontPtsCnt, d.z + fontPtsCnt, d.y + fontPtsCnt);
		}
		for (const auto&b: borders)
		{
			const int& i_0 = b.x;
			const int& i_1 = b.y;
			const int& i_2 = b.x+ fontPtsCnt;
			const int& i_3 = b.y+ fontPtsCnt;
			faces.emplace_back(i_0, i_2, i_1);
			faces.emplace_back(i_1, i_2, i_3);
		}  
		std::fstream fout("../surf/p.obj", std::ios::out);
		for (int i = 0; i < pts.size(); i++)
		{
			fout << "v " << pts[i].x << " " << pts[i].y << " " << pts[i].z << std::endl;
		}
		for (int i = 0; i < faces.size(); i++)
		{
			fout << "f " << faces[i].x+1 << " " << faces[i].y + 1 << " " << faces[i].z + 1 << std::endl;
		}
		fout.close();
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
			if (imgDirs.empty())
			{
				float fxInv = 1. / fx;
				float fyInv = 1. / fy;
				imgDirs = cv::Mat::zeros(height, width, CV_32FC3);
				for (int r = 0; r < height; r++)
				{
					for (int c = 0; c < width; c++)
					{
						Eigen::Vector3f dir((c + 0.5 - cx) * fxInv, (r + 0.5 - cy) * fyInv, 1);
						dir.normalize();
						imgDirs.at<cv::Vec3f>(r, c)[0] = dir.x();
						imgDirs.at<cv::Vec3f>(r, c)[1] = dir.y();
						imgDirs.at<cv::Vec3f>(r, c)[2] = dir.z();
					}
				}
			}
		}
		std::vector<char>serialization()const
		{
			std::vector<char>ret(6 * sizeof(float));
			float* data = (float*)&ret[0];
			data[0] = fx;
			data[1] = fy;
			data[2] = cx;
			data[3] = cy;
			data[4] = height;
			data[5] = width;
			return ret;
		}
		int deserialization(const char* data)
		{
			const float* floatData = (float*)&data[0];
			fx = floatData[0];
			fy = floatData[1];
			cx = floatData[2];
			cy = floatData[3];
			height = floatData[4];
			width = floatData[5];
			intr = Eigen::Matrix4f::Identity();
			intr(0, 0) = fx;
			intr(1, 1) = fy;
			intr(0, 2) = cx;
			intr(1, 2) = cy;
			if (imgDirs.empty())
			{
				float fxInv = 1. / fx;
				float fyInv = 1. / fy;
				imgDirs = cv::Mat::zeros(height, width, CV_32FC3);
				for (int r = 0; r < height; r++)
				{
					for (int c = 0; c < width; c++)
					{
						Eigen::Vector3f dir((c + 0.5 - cx) * fxInv, (r + 0.5 - cy) * fyInv, 1);
						dir.normalize();
						imgDirs.at<cv::Vec3f>(r, c)[0] = dir.x();
						imgDirs.at<cv::Vec3f>(r, c)[1] = dir.y();
						imgDirs.at<cv::Vec3f>(r, c)[2] = dir.z();
					}
				}
			}
			return 6 * sizeof(float);
		}
		static std::vector<char>serialization(const std::unordered_map<int, Camera>& cameras)
		{
			std::vector<char>ret(sizeof(int));;
			*(int*)&ret[0] = cameras.size();
			for (const auto& d : cameras)
			{
				std::vector<char>ret0(sizeof(int));
				*(int*)&ret0[0] = d.first;
				std::vector<char>ret1 = d.second.serialization();
				ret.insert(ret.end(), ret0.begin(), ret0.end());
				ret.insert(ret.end(), ret1.begin(), ret1.end());
			}
			return ret;
		}
		static int deserialization(const char* data, std::unordered_map<int, Camera>& dataOut)
		{
			dataOut.clear();
			int elemCnt = *(int*)data;
			int pos = sizeof(int);
			for (int i = 0; i < elemCnt; i++)
			{
				int cameraId = *(int*)&data[pos]; pos += sizeof(int);
				dataOut[cameraId] = Camera();
				int deserialLength = dataOut[cameraId].deserialization(&data[pos]); pos += deserialLength;
			}
			return pos;
		}
		static int figureCameraHash(const Camera& c)
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
		cv::Mat getImgDirs()const
		{
			return imgDirs;
		}
		double fx, fy, cx, cy;
		int height, width;
		Eigen::Matrix4f intr;
		cv::Mat imgDirs;
	};
	struct View
	{
		int viewId;
		int cameraId;
		Eigen::Matrix4f Rt;
		std::filesystem::path imgPath;
		std::filesystem::path maskPath;
		std::unordered_map<std::uint32_t, std::vector<std::uint32_t>>pixelGridBelong;//thisView    imgPixelId:gridsId 
		std::vector<char>serialization()const
		{
			std::vector<char>sndata(sizeof(int)*2);
			*(int*)&sndata[0] = viewId;
			*(int*)&sndata[sizeof(int)] = cameraId;
			std::vector<char>Rtdata(16 * sizeof(float));
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
			for (const auto& d : pixelGridBelong)
			{
				pixelGridBelongData.emplace_back(d.first);
				pixelGridBelongData.emplace_back(d.second.size());
				for (const auto& d2 : d.second)
				{
					pixelGridBelongData.emplace_back(d2);
				}
			}
			std::vector<char>pixelGridBelongData_(pixelGridBelongData.size() * sizeof(std::uint32_t));
			std::uint32_t* data = (std::uint32_t*)&pixelGridBelongData_[0];
			for (const auto& d : pixelGridBelongData)
			{
				*data = d;
				data++;
			}
			sndata.insert(sndata.end(), Rtdata.begin(), Rtdata.end());
			sndata.insert(sndata.end(), imgPathData.begin(), imgPathData.end());
			sndata.insert(sndata.end(), maskPathData.begin(), maskPathData.end());
			sndata.insert(sndata.end(), pixelGridBelongData_.begin(), pixelGridBelongData_.end());
			return sndata;
		}
		int deserialization(const char* data)
		{
			int pos = 0;
			viewId = *(int*)data;
			pos += sizeof(int);
			cameraId = *(int*)(&data[sizeof(int)]);
			pos += sizeof(int);
			const float* Rtdata_ = (float*)&data[pos];
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
				pixelGridBelong[key_].resize(memberSize_);
				for (int j = 0; j < memberSize_; j++)
				{
					pixelGridBelong[key_][j] = intData[i]; i++;
				}
			}
			return pos + sizeof(std::uint32_t) * i;
		}
		static std::vector<char>serialization(const std::unordered_map<int, View>& views)
		{
			std::vector<char>ret(sizeof(int));;
			*(int*)&ret[0] = views.size();
			for (const auto& d : views)
			{
				std::vector<char>ret0(sizeof(int));
				*(int*)&ret0[0] = d.first;
				std::vector<char>ret1 = d.second.serialization();
				ret.insert(ret.end(), ret0.begin(), ret0.end());
				ret.insert(ret.end(), ret1.begin(), ret1.end());
			}
			return ret;
		}
		static int deserialization(const char* data, std::unordered_map<int, View>& dataOut)
		{
			dataOut.clear();
			int elemCnt = *(int*)data;
			int pos = sizeof(int);
			for (int i = 0; i < elemCnt; i++)
			{
				int viewId = *(int*)&data[pos]; pos += sizeof(int);
				dataOut[viewId] = View();
				int deserialLength = dataOut[viewId].deserialization(&data[pos]); pos += deserialLength;
			}
			return pos;
		}
		cv::Mat getImgWorldDir(const Camera& c)
		{
			if (this->imgWorldDir.empty())
			{
				const cv::Mat& imgDir = c.getImgDirs();
				const Eigen::Matrix4f Rtinv = this->Rt.inverse();
				imgWorldDir = cv::Mat::zeros(imgDir.size(), CV_32FC3);
				Eigen::Matrix4Xf A(4, imgWorldDir.rows * imgWorldDir.cols);
				for (int i = 0; i < A.cols(); i++)
				{
					int r = i / imgWorldDir.cols;
					int c = i % imgWorldDir.cols;
					const cv::Vec3f& dir = imgDir.at<cv::Vec3f>(r, c);
					A(0, i) = dir[0];
					A(1, i) = dir[1];
					A(2, i) = dir[2];
					A(3, i) = 1.;
				}
				Eigen::Matrix4Xf B = Rtinv * A;
				for (int i = 0; i < A.cols(); i++)
				{
					int r = i / imgWorldDir.cols;
					int c = i % imgWorldDir.cols;
					imgWorldDir.at<cv::Vec3f>(r, c)[0] = B(0, i);
					imgWorldDir.at<cv::Vec3f>(r, c)[1] = B(1, i);
					imgWorldDir.at<cv::Vec3f>(r, c)[2] = B(2, i);
				}
			}
			return this->imgWorldDir;
		}
		cv::Mat getImgWorldDir()const
		{
			return this->imgWorldDir;
		}
		cv::Mat imgWorldDir;
	};

	class Mrf
	{
#define XyzToGridXYZ(x,y,z,xStart,yStart,zStart,gridUnitInv,gridX,gridY,gridZ) \
		int gridX = (x-xStart)*gridUnitInv;  \
		int gridY = (y-yStart)*gridUnitInv;  \
		int gridZ = (z-zStart)*gridUnitInv;  
#define GridXyzToPosEncode(gridX,gridY,gridZ,resolutionX,resolutionXY,posEncode) \
		std::uint32_t posEncode = gridX+gridY*resolutionX+gridZ*resolutionXY;
#define PosEncodeToGridXyz(posEncode,resolutionX,resolutionXY,gridX,gridY,gridZ) \
		std::uint32_t gridZ = posEncode / resolutionXY;							\
		std::uint32_t gridY = posEncode % resolutionXY;							\
		std::uint32_t gridX = gridY % resolutionX;								\
		gridY = gridY / resolutionX;
#define ImgXyToImgPixelEncode(x,y,imgWidth)  (imgWidth*y+x)
#define ImgPixelEncodeToImgXy(imgPixelEncode,x,y,imgWidth)  \
		int y = imgPixelEncode / imgWidth;				\
		int x = imgPixelEncode % imgWidth;

	public:
		Mrf()
		{}
		Mrf(const std::filesystem::path& dataDir_, const std::filesystem::path& objPath_, const GridConfig& gridConfig_)
		{
			dataDir = dataDir_;
			objPath = objPath_;
			gridConfig = gridConfig_;
			this->gridResolution= gridConfig.gridResolution;
			this->gridResolutionInv = 1. / this->gridResolution;;
			{
				bool loadObj = mrf::readObj(objPath, this->vertex, this->faces);
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

				this->resolutionX = (this->targetMaxBorderX - this->targetMinBorderX) / gridConfig.gridResolution + 1;
				this->resolutionY = (this->targetMaxBorderY - this->targetMinBorderY) / gridConfig.gridResolution + 1;
				this->resolutionZ = (this->targetMaxBorderZ - this->targetMinBorderZ) / gridConfig.gridResolution + 1;
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
						std::string shortName__ = thisFilename.filename().stem().string();
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
						int thisCameraSn = Camera::figureCameraHash(thisCamera);
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
						if (std::filesystem::exists(imgPath)&& std::filesystem::exists(maskPath))
						{
							int viewIdx = this->views.size();
							View thisView = { viewIdx,thisCameraSn, Rt ,imgPath ,maskPath };							
							this->views[viewIdx] = thisView;
						}
						else
						{
							LOG_ERR_OUT << "imgPath or maskPath not exists : "<< imgPath<<" or "<< maskPath;
							continue;
						}
					}
				}
			}
			for (auto&view : this->views)
			{
				const int& viewIdx = view.first;
				View& thisView = view.second;
				const int& thisCameraSn = thisView.cameraId;
				const Camera& thisCamera = this->cameras[thisCameraSn];
				int height = thisCamera.height;
				int width = thisCamera.width;
				const Eigen::Matrix4f& Rt = thisView.Rt;
				cv::Mat img = cv::imread(thisView.imgPath.string());
				cv::Mat mask = tools::loadMask(thisView.maskPath.string());
				if (img.rows != mask.rows || img.cols != mask.cols)
				{
					LOG_ERR_OUT << "img.rows != mask.rows || img.cols != mask.cols";
					exit(-1);
				}
				cv::Mat rayDistance = cv::Mat::ones(mask.size(), CV_32FC1) * -1;
				const Eigen::Matrix3f R = Rt.block(0, 0, 3, 3);
				const Eigen::Matrix4f KRt = thisCamera.intr * Rt;
				Eigen::Vector3f cameraT(Rt(0, 3), Rt(1, 3), Rt(2, 3));
				std::vector<cv::Vec2i>vertexInImg(this->vertex.cols());
				std::vector<float>vertexDist(this->vertex.cols());
				std::vector<bool> vertexInViewInCanvas(this->vertex.cols(), false);
				{
					Eigen::Matrix4Xf vertexInView = Rt * this->vertex;
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
						const auto& a = this->faces(0, i);
						const auto& b = this->faces(1, i);
						const auto& c = this->faces(2, i);
						if (vertexInViewInCanvas[a] && vertexInViewInCanvas[b] && vertexInViewInCanvas[c])
						{
							const cv::Vec2i& uv0 = vertexInImg[a];
							const cv::Vec2i& uv1 = vertexInImg[b];
							const cv::Vec2i& uv2 = vertexInImg[c];
							int minX = (std::min)((std::min)(uv0[0], uv1[0]), uv2[0]);
							int minY = (std::min)((std::min)(uv0[1], uv1[1]), uv2[1]);
							int maxX = (std::max)((std::max)(uv0[0], uv1[0]), uv2[0]);
							int maxY = (std::max)((std::max)(uv0[1], uv1[1]), uv2[1]);
							float nearestDist = (std::min)((std::min)(vertexDist[a], vertexDist[b]), vertexDist[c]);
							for (int r_ = minY; r_ <= maxY; r_++)
							{
								for (int c_ = minX; c_ <= maxX; c_++)
								{
									if (mask.ptr<uchar>(r_)[c_] > 0)
									{
										float& dist = rayDistance.ptr<float>(r_)[c_];
										if (dist<0 || dist>nearestDist)
										{
											dist = nearestDist;
										}
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
				this->registerRayDistanceMap(rayDistance, thisCamera, thisView);
			}

		}
		bool registerRayDistanceMap(const cv::Mat& rayDistantMap, const Camera& camera, View& view)
		{
			LOG_OUT << view.imgPath;
			cv::Mat img = cv::imread(view.imgPath.string());
			cv::cvtColor(img, img, cv::COLOR_BGR2RGB);
			std::unordered_map<std::uint32_t, std::unordered_set<std::uint32_t>>pixelGridBelong_map;
			cv::Mat imgDirs = camera.getImgDirs();
			const Eigen::Matrix4f Rtinv = view.Rt.inverse();
			const Eigen::Matrix3f Rinv = Rtinv.block(0, 0, 3, 3);
			const Eigen::Vector3f tinv(Rtinv(0, 3), Rtinv(1, 3), Rtinv(2, 3));
			const cv::Point3f cameraCenter(view.Rt(0, 3), view.Rt(1, 3), view.Rt(2, 3));
			float rayStepUnit = this->gridConfig.gridResolution*0.5;
			int amplitudeCnt = this->gridConfig.rayPixelCnt;
			{
				for (int r = 0; r < camera.height; r++)
				{
					for (int c = 0; c < camera.width; c++)
					{
						const float& rayDist = rayDistantMap.ptr<float>(r)[c];
						if (1e-3 < rayDist)
						{
							Eigen::Vector3f pixelDir = Eigen::Vector3f(imgDirs.at<cv::Vec3f>(r, c)[0], imgDirs.at<cv::Vec3f>(r, c)[1], imgDirs.at<cv::Vec3f>(r, c)[2]);
							std::uint32_t imgPixelPos = ImgXyToImgPixelEncode(c, r, camera.width);
							for (int i = 0;; i++)
							{
								if (i>1000)
								{
									LOG_ERR_OUT << "i>1000 : r=" << r << "; c=" << c;
									exit(-1);
								}
								{
									Eigen::Vector3f p0 = (rayDist - i * rayStepUnit) * pixelDir;
									Eigen::Vector4f p = Rtinv * Eigen::Vector4f(p0[0], p0[1], p0[2], 1);
									if (p[0] < this->targetMinBorderX || p[1] < this->targetMinBorderY || p[2] < this->targetMinBorderZ
										|| p[0] >= this->targetMaxBorderX || p[1] >= this->targetMaxBorderY || p[2] >= this->targetMaxBorderZ)
									{
										continue;
									}
									XyzToGridXYZ(p[0], p[1], p[2], this->targetMinBorderX, this->targetMinBorderY, this->targetMinBorderZ, this->gridResolutionInv, gridX, gridY, gridZ);
									GridXyzToPosEncode(gridX, gridY, gridZ, this->resolutionX, this->resolutionXY, posEncode);
									pixelGridBelong_map[imgPixelPos].emplace(posEncode);
								}
								if (pixelGridBelong_map[imgPixelPos].size() >= amplitudeCnt)
								{
									break;
								}
								{
									Eigen::Vector3f p0 = (rayDist + i * rayStepUnit) * pixelDir;
									Eigen::Vector4f p = Rtinv * Eigen::Vector4f(p0[0], p0[1], p0[2], 1);
									if (p[0] < this->targetMinBorderX || p[1] < this->targetMinBorderY || p[2] < this->targetMinBorderZ
										|| p[0] >= this->targetMaxBorderX || p[1] >= this->targetMaxBorderY || p[2] >= this->targetMaxBorderZ)
									{
										continue;
									}
									XyzToGridXYZ(p[0], p[1], p[2], this->targetMinBorderX, this->targetMinBorderY, this->targetMinBorderZ, this->gridResolutionInv, gridX, gridY, gridZ);
									GridXyzToPosEncode(gridX, gridY, gridZ, this->resolutionX, this->resolutionXY, posEncode);
									pixelGridBelong_map[imgPixelPos].emplace(posEncode);
								}
								if (pixelGridBelong_map[imgPixelPos].size() >= amplitudeCnt)
								{
									break;
								}
							}
						}
					}
				}
			}
			for (const auto& d : pixelGridBelong_map)
			{
				const int& imgPixelId = d.first;
				ImgPixelEncodeToImgXy(imgPixelId, x, y, camera.width);
				const int elemSize = d.second.size();
				std::vector<std::pair<float, std::uint32_t>>sortWithDists(elemSize);
				int i = 0;
				for (const auto& d2 : d.second)
				{
					sortWithDists[i].second = d2;
					PosEncodeToGridXyz(d2, this->resolutionX, this->resolutionXY, gridX, gridY, gridZ);
					cv::Point3f gridCentor;
					gridCentor.x = (0.5+gridX) * this->gridResolution + this->targetMinBorderX;
					gridCentor.y = (0.5+gridY) * this->gridResolution + this->targetMinBorderY;
					gridCentor.z = (0.5+gridZ) * this->gridResolution + this->targetMinBorderZ;
					sortWithDists[i].first = cv::norm(cameraCenter - gridCentor);
					i++;
				}
				std::sort(sortWithDists.begin(), sortWithDists.end(), [](const auto& a, const auto& b) {return a.first < b.first; });
				view.pixelGridBelong[d.first].clear();
				view.pixelGridBelong[d.first].resize(elemSize);
				for (i = 0; i < elemSize; i++)
				{
					const uint32_t& gridId = sortWithDists[i].second;
					view.pixelGridBelong[d.first][i] = gridId;
					if (gridInViews.count(gridId)==0)
					{
						gridInViews[gridId] = std::vector<std::uint8_t>(this->views.size(), 0);
						gridRgbset[gridId] = std::vector<cv::Vec3b>(this->views.size(), cv::Vec3b(0,0,0));
					}
					gridInViews[gridId][view.viewId] = 1;
					gridRgbset[gridId][view.viewId] = img.at<cv::Vec3b>(y,x);
				}
			}
			return true;
		}
		static Eigen::MatrixXf rayDistantMapToPts(const cv::Mat& rayDistantMap, const Camera& camera, const Eigen::Matrix4f& viewRt)
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
				for (const auto& d : listRet)
				{
					ret(0, i) = d.x();
					ret(1, i) = d.y();
					ret(2, i) = d.z();
					ret(3, i) = 1;
					i += 1;
				}
			}
			const Eigen::Matrix4f Rtinv = viewRt.inverse();
			return Rtinv * ret;
		}
		~Mrf()
		{}
		std::vector<char>serialization()const
		{
			std::vector<char>ret;
			std::vector<char>imgPathData, maskPathData, vertexData, facesData;
			transform(this->dataDir.string(), imgPathData);
			transform(this->objPath.string(), maskPathData);
			transform(this->vertex, vertexData);
			transform(this->faces, facesData);
			ret.insert(ret.end(), imgPathData.begin(), imgPathData.end());
			ret.insert(ret.end(), maskPathData.begin(), maskPathData.end());
			ret.insert(ret.end(), vertexData.begin(), vertexData.end());
			ret.insert(ret.end(), facesData.begin(), facesData.end());
			{
				std::vector<char>floatDat(8 * sizeof(float));
				*(float*)&floatDat[0 * sizeof(float)] = targetMinBorderX;
				*(float*)&floatDat[1 * sizeof(float)] = targetMinBorderY;
				*(float*)&floatDat[2 * sizeof(float)] = targetMinBorderZ;
				*(float*)&floatDat[3 * sizeof(float)] = targetMaxBorderX;
				*(float*)&floatDat[4 * sizeof(float)] = targetMaxBorderY;
				*(float*)&floatDat[5 * sizeof(float)] = targetMaxBorderZ;
				*(float*)&floatDat[6 * sizeof(float)] = gridResolution;
				*(float*)&floatDat[7 * sizeof(float)] = gridResolutionInv;
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
			std::vector<char> gridInViewsAndRgbset(sizeof(std::uint32_t)+gridInViews.size()*(sizeof(std::uint32_t)+ this->views.size()*(sizeof(std::uint8_t)+sizeof(cv::Vec3b))));
			int pos = 0;
			*(std::uint32_t*)&gridInViewsAndRgbset[pos] = gridInViews.size();
			pos += sizeof(std::uint32_t);
			for (const auto&d: gridInViews)
			{
				const std::uint32_t& gridId = d.first;
				*(std::uint32_t*)&gridInViewsAndRgbset[pos] = gridId;
				pos += sizeof(std::uint32_t);
				for (int i = 0; i < gridInViews.at(gridId).size(); i++)
				{
					*(std::uint8_t*)&gridInViewsAndRgbset[pos] = gridInViews.at(gridId)[i];
					pos += sizeof(std::uint8_t);
				}
				for (int i = 0; i < gridRgbset.at(gridId).size(); i++)
				{
					*(uchar*)&gridInViewsAndRgbset[pos] = gridRgbset.at(gridId)[i][0];
					pos += sizeof(uchar);
					*(uchar*)&gridInViewsAndRgbset[pos] = gridRgbset.at(gridId)[i][1];
					pos += sizeof(uchar);
					*(uchar*)&gridInViewsAndRgbset[pos] = gridRgbset.at(gridId)[i][2];
					pos += sizeof(uchar);
				}
			}
			ret.insert(ret.end(), gridInViewsAndRgbset.begin(), gridInViewsAndRgbset.end());
			return ret;
		}
		bool serialization(const std::filesystem::path&path)const
		{
			std::vector<char>d = this->serialization();
			std::fstream fout(path,std::ios::out | std::ios::binary);
			int dCnt = d.size();
			fout.write((char*)&dCnt, sizeof(int));
			fout.write(&d[0], dCnt * sizeof(char));
			fout.close();
			return true;
		}
		bool reload(const std::filesystem::path& path)
		{
			std::fstream fin(path, std::ios::in | std::ios::binary);
			int totalSize = 0;
			fin.read((char*)&totalSize, sizeof(int));
			std::vector<char>dat(totalSize);
			fin.read((char*)&dat[0], totalSize * sizeof(char));
			int pos = 0;
			std::string line;
			pos += transform(&dat[pos], line); this->dataDir = std::filesystem::path(line);
			pos += transform(&dat[pos], line); this->objPath = std::filesystem::path(line);
			pos += transform(&dat[pos], this->vertex);
			pos += transform(&dat[pos], this->faces);
			{
				targetMinBorderX = *(float*)&dat[pos]; pos += sizeof(float);
				targetMinBorderY = *(float*)&dat[pos]; pos += sizeof(float);
				targetMinBorderZ = *(float*)&dat[pos]; pos += sizeof(float);
				targetMaxBorderX = *(float*)&dat[pos]; pos += sizeof(float);
				targetMaxBorderY = *(float*)&dat[pos]; pos += sizeof(float);
				targetMaxBorderZ = *(float*)&dat[pos]; pos += sizeof(float);
				gridResolution = *(float*)&dat[pos]; pos += sizeof(float);
				gridResolutionInv = *(float*)&dat[pos]; pos += sizeof(float);
				resolutionX = *(int*)&dat[pos]; pos += sizeof(int);
				resolutionY = *(int*)&dat[pos]; pos += sizeof(int);
				resolutionZ = *(int*)&dat[pos]; pos += sizeof(int);
				resolutionXY = *(int*)&dat[pos]; pos += sizeof(int);
			}
			pos += gridConfig.deserialization(&dat[pos]);
			pos += Camera::deserialization(&dat[pos], this->cameras);
			pos += View::deserialization(&dat[pos], this->views);
			for (auto& v : this->views)
			{
				const auto& cameraId = v.second.cameraId;
				auto& thisCamera = this->cameras.at(cameraId);
				v.second.getImgWorldDir(thisCamera);
			}

			std::uint32_t gridInViewsCnt = *(std::uint32_t*)&dat[pos];
			pos += sizeof(std::uint32_t);
			for (int i = 0; i < gridInViewsCnt; i++)
			{
				std::uint32_t gridId = *(std::uint32_t*)&dat[pos];
				pos += sizeof(std::uint32_t);
				this->gridInViews[gridId].resize(this->views.size());
				this->gridRgbset[gridId].resize(this->views.size());
				for (int j = 0; j < this->views.size(); j++)
				{
					this->gridInViews[gridId][j] = *(std::uint8_t*)&dat[pos];
					pos += sizeof(std::uint8_t);
				}
				for (int j = 0; j < this->views.size(); j++)
				{
					this->gridRgbset[gridId][j][0] = *(uchar*)&dat[pos];
					pos += sizeof(uchar);
					this->gridRgbset[gridId][j][1] = *(uchar*)&dat[pos];
					pos += sizeof(uchar);
					this->gridRgbset[gridId][j][2] = *(uchar*)&dat[pos];
					pos += sizeof(uchar);
				}
			}
			return true;
		}
		bool saveGrid(const std::filesystem::path& path, std::unordered_map<std::uint32_t, float>& gridScores)
		{
			std::vector<std::uint32_t>gridIds(gridScores.size());
			std::vector<float>scores(gridScores.size());
			std::uint32_t i = 0;
			for (const auto& d : gridScores)
			{
				gridIds[i] = d.first;
				scores[i] = d.second;
				++i;
			}
			std::fstream fout(path, std::ios::out | std::ios::binary);
			fout.write((char*)&i, sizeof(std::uint32_t));
			fout.write((char*)&gridIds[0], sizeof(std::uint32_t) * gridIds.size());
			fout.write((char*)&scores[0], sizeof(float) * gridIds.size());
			fout.close();
			std::fstream fPtsout(path.parent_path()/"123.txt", std::ios::out);
			for (const auto& d : gridScores)
			{
				PosEncodeToGridXyz(d.first, this->resolutionX, this->resolutionXY, gridX, gridY, gridZ);
				float x = (0.5 + gridX) * this->gridResolution + this->targetMinBorderX;
				float y = (0.5 + gridY) * this->gridResolution + this->targetMinBorderY;
				float z = (0.5 + gridZ) * this->gridResolution + this->targetMinBorderZ;
				fPtsout << x << " " << y << " " << z <<" "<<d.second << std::endl;
			}
			fPtsout.close();
			return true;
		}
		int figurePixelColorScore(const std::function<float(const std::vector<std::uint8_t>& inWhichViews, const std::vector<cv::Vec3b>& rgbSet)>&scoreFun)const
		{			
			std::unordered_map<std::uint32_t, float>gridColorScore;
			for (const auto&d:this->gridInViews)
			{
				const auto& gridId = d.first;
				const std::vector<std::uint8_t>& inWhichViews = gridInViews.at(gridId);
				const std::vector<cv::Vec3b>& rgbSet = gridRgbset.at(gridId);
				gridColorScore[gridId] = scoreFun(inWhichViews, rgbSet);
			}
			std::unordered_set<std::uint32_t>pickedMaxColorScoreGrids;
			for (auto& v : this->views)
			{
				const auto& cameraId = v.second.cameraId;
				auto& thisCamera = this->cameras.at(cameraId);
				const auto& thisView = v.second;
				for (const auto& pixel : thisView.pixelGridBelong)
				{
					const std::uint32_t& pixelId = pixel.first;
					const std::vector<std::uint32_t>& gridsId = pixel.second;					
					std::uint32_t maxColorScoreGrid = gridsId[0];
					float maxColorScoreGridColorScore = gridColorScore[maxColorScoreGrid];
					for (size_t i = 1; i < gridsId.size(); i++)
					{
						const std::uint32_t& thisGridId = gridsId[i];
						const float& thisGridColorScore = gridColorScore[thisGridId];
						if (thisGridColorScore> maxColorScoreGridColorScore)
						{
							maxColorScoreGridColorScore = thisGridColorScore;
							maxColorScoreGrid = thisGridId;
						}
					}
					pickedMaxColorScoreGrids.insert(maxColorScoreGrid);
				}
			}
			std::fstream fout("../surf/123.txt", std::ios::out);
			for (const auto&d: pickedMaxColorScoreGrids)
			{
				PosEncodeToGridXyz(d, this->resolutionX, this->resolutionXY, gridX, gridY, gridZ);
				float x = (0.5 + gridX) * this->gridResolution + this->targetMinBorderX;
				float y = (0.5 + gridY) * this->gridResolution + this->targetMinBorderY;
				float z = (0.5 + gridZ) * this->gridResolution + this->targetMinBorderZ;
				fout << x << " " << y << " " << z << std::endl;
			}
			fout.close();
			return 0;
		} 
		int findNeightDist(std::unordered_map<std::uint32_t, std::unordered_map<std::uint32_t, unsigned char>>&neightDist, const int& distThre)const
		{
			if (distThre>3)
			{
				LOG_ERR_OUT << "not support";
				return -1;
			}
			std::vector<int> shifts;
			std::vector<unsigned char> shiftDists;
			shifts.reserve(32);
			shiftDists.reserve(32);
			switch (distThre)
			{
			case 1:
				shifts.emplace_back(-this->resolutionXY);	   shiftDists.emplace_back(1);
				shifts.emplace_back(-1);					   shiftDists.emplace_back(1);
				shifts.emplace_back(-this->resolutionX);	   shiftDists.emplace_back(1);
				shifts.emplace_back(1);						   shiftDists.emplace_back(1);
				shifts.emplace_back(this->resolutionX);		   shiftDists.emplace_back(1);
				shifts.emplace_back(this->resolutionXY);	   shiftDists.emplace_back(1);
				break;
			case 2:
				shifts.emplace_back(-this->resolutionXY - this->resolutionX);	  shiftDists.emplace_back(3-2);
				shifts.emplace_back(-this->resolutionXY - 1);					  shiftDists.emplace_back(3-2);
				shifts.emplace_back(-this->resolutionXY);						  shiftDists.emplace_back(3-1);
				shifts.emplace_back(-this->resolutionXY + 1);					  shiftDists.emplace_back(3-2);
				shifts.emplace_back(-this->resolutionXY + this->resolutionX);	  shiftDists.emplace_back(3-2);
				shifts.emplace_back(-1 - this->resolutionX);					  shiftDists.emplace_back(3-2);
				shifts.emplace_back(-this->resolutionX);						  shiftDists.emplace_back(3-1);
				shifts.emplace_back(-this->resolutionX + 1);					  shiftDists.emplace_back(3-2);
				shifts.emplace_back(-1);										  shiftDists.emplace_back(3-1);
				shifts.emplace_back(1);											  shiftDists.emplace_back(3-1);
				shifts.emplace_back(-1 + this->resolutionX);					  shiftDists.emplace_back(3-2);
				shifts.emplace_back(this->resolutionX);							  shiftDists.emplace_back(3-1);
				shifts.emplace_back(this->resolutionX + 1);						  shiftDists.emplace_back(3-2);
				shifts.emplace_back(this->resolutionXY - this->resolutionX);	  shiftDists.emplace_back(3-2);
				shifts.emplace_back(this->resolutionXY - 1);					  shiftDists.emplace_back(3-2);
				shifts.emplace_back(this->resolutionXY);						  shiftDists.emplace_back(3-1);
				shifts.emplace_back(this->resolutionXY + 1);					  shiftDists.emplace_back(3-2);
				shifts.emplace_back(this->resolutionXY + this->resolutionX);	  shiftDists.emplace_back(3-2);
				break;
			case 3:
				shifts.emplace_back(-this->resolutionXY - 1 - this->resolutionX);	shiftDists.emplace_back(4-3);
				shifts.emplace_back(-this->resolutionXY - this->resolutionX);		shiftDists.emplace_back(4-2);
				shifts.emplace_back(-this->resolutionXY - this->resolutionX + 1);	shiftDists.emplace_back(4-3);
				shifts.emplace_back(-this->resolutionXY - 1);						shiftDists.emplace_back(4-2);
				shifts.emplace_back(-this->resolutionXY);							shiftDists.emplace_back(4-1);
				shifts.emplace_back(-this->resolutionXY + 1);						shiftDists.emplace_back(4-2);
				shifts.emplace_back(-this->resolutionXY - 1 + this->resolutionX);	shiftDists.emplace_back(4-3);
				shifts.emplace_back(-this->resolutionXY + this->resolutionX);		shiftDists.emplace_back(4-2);
				shifts.emplace_back(-this->resolutionXY + this->resolutionX + 1);	shiftDists.emplace_back(4-3);
				shifts.emplace_back(-1 - this->resolutionX);						shiftDists.emplace_back(4-2);
				shifts.emplace_back(-this->resolutionX);							shiftDists.emplace_back(4-1);
				shifts.emplace_back(-this->resolutionX + 1);						shiftDists.emplace_back(4-2);
				shifts.emplace_back(-1);											shiftDists.emplace_back(4-1);
				shifts.emplace_back(1);												shiftDists.emplace_back(4-1);
				shifts.emplace_back(-1 + this->resolutionX);						shiftDists.emplace_back(4-2);
				shifts.emplace_back(this->resolutionX);								shiftDists.emplace_back(4-1);
				shifts.emplace_back(this->resolutionX + 1);							shiftDists.emplace_back(4-2);
				shifts.emplace_back(this->resolutionXY - 1 - this->resolutionX);	shiftDists.emplace_back(4-3);
				shifts.emplace_back(this->resolutionXY - this->resolutionX);		shiftDists.emplace_back(4-2);
				shifts.emplace_back(this->resolutionXY - this->resolutionX + 1);	shiftDists.emplace_back(4-3);
				shifts.emplace_back(this->resolutionXY - 1);						shiftDists.emplace_back(4-2);
				shifts.emplace_back(this->resolutionXY);							shiftDists.emplace_back(4-1);
				shifts.emplace_back(this->resolutionXY + 1);						shiftDists.emplace_back(4-2);
				shifts.emplace_back(this->resolutionXY - 1 + this->resolutionX);	shiftDists.emplace_back(4-3);
				shifts.emplace_back(this->resolutionXY + this->resolutionX);		shiftDists.emplace_back(4-2);
				shifts.emplace_back(this->resolutionXY + this->resolutionX + 1);	shiftDists.emplace_back(4-3);
				break;
			default:
				break;
			}


			for (const auto&d:this->gridInViews)
			{
				const std::uint32_t& thisGridId = d.first;
				for (int i = 0; i < shifts.size(); i++)
				{
					std::uint32_t neightId = thisGridId;
					if (shifts[i]<0 && thisGridId> abs(shifts[i]))
					{
						neightId = thisGridId - abs(shifts[i]);
					}
					else if (shifts[i]>= 0 && thisGridId< thisGridId + shifts[i])
					{
						neightId = thisGridId + shifts[i];
					} 
					if (thisGridId!= neightId)
					{
						neightDist[thisGridId][neightId] = shiftDists[i];
					}
				}
			}



			return 0;
		}
		int mrf1()
		{
			std::filesystem::path outDir = "../surf";
			const auto& scoreFun = mrf::Mrf::getColorScore;
			std::unordered_map<std::uint32_t, float>gridColorScore;
			for (const auto& d : this->gridInViews)
			{
				const auto& gridId = d.first;
				const std::vector<std::uint8_t>& inWhichViews = gridInViews.at(gridId);
				const std::vector<cv::Vec3b>& rgbSet = gridRgbset.at(gridId);
				gridColorScore[gridId] = scoreFun(inWhichViews, rgbSet);
			}
 

			std::unordered_map<std::uint32_t, std::unordered_map<std::uint32_t, unsigned char>> neightDist;
			findNeightDist(neightDist,3);
			std::unordered_map<std::uint32_t, float>gridAccumScore;
			{
				//init
				for (const auto&d: neightDist)
				{
					gridAccumScore[d.first] = 0.;
				}
			}
			int iterCnt = 20;
			for (int iter = 0; iter < iterCnt; iter++)
			{
				LOG_OUT << "iter = " << iter;
				for (auto& v : this->views)
				{
					const auto& cameraId = v.second.cameraId;
					auto& thisCamera = this->cameras.at(cameraId);
					const auto& thisView = v.second;
					for (const auto& pixel : thisView.pixelGridBelong)
					{
						const std::uint32_t& pixelId = pixel.first;
						const std::vector<std::uint32_t>& gridsId = pixel.second;
						std::uint32_t maxColorScoreGrid = gridsId[0];
						float maxColorScoreGridColorScore = gridColorScore[maxColorScoreGrid] + gridAccumScore[maxColorScoreGrid];
						for (size_t i = 1; i < gridsId.size(); i++)
						{
							const std::uint32_t& thisGridId = gridsId[i];
							const float& thisGridColorScore = gridColorScore[thisGridId] + gridAccumScore[thisGridId];
							if (thisGridColorScore > maxColorScoreGridColorScore)
							{
								maxColorScoreGridColorScore = thisGridColorScore;
								maxColorScoreGrid = thisGridId;
							}
						}
						const std::unordered_map<std::uint32_t, unsigned char>& neighbs = neightDist[maxColorScoreGrid];
						for (const auto& neighb: neighbs)
						{
							if (thisView.pixelGridBelong.find(neighb.first)== thisView.pixelGridBelong.end())
							{
								gridAccumScore[maxColorScoreGrid] += neighb.second * 0.03* gridColorScore[neighb.first];
							}
						}
					}
				}
			}
			for (const auto& d : neightDist)
			{
				gridAccumScore[d.first] += gridColorScore[d.first];
			}



			
			for (const auto& v : this->views)
			{				
				const auto& cameraId = v.second.cameraId;
				auto& thisCamera = this->cameras.at(cameraId);
				const auto& thisView = v.second;
				cv::Mat mask = tools::loadMask(thisView.maskPath.string());
				cv::Mat ptsMatMask = cv::Mat::zeros(mask.size(), CV_8UC1);
				cv::Mat ptsMat = cv::Mat::zeros(mask.size(), CV_32FC3);
				//std::fstream fout1(outDir / (std::to_string(thisView.viewId) + ".txt"), std::ios::out);
				for (const auto& pixel : thisView.pixelGridBelong)
				{
					const std::uint32_t& pixelId = pixel.first;
					const std::vector<std::uint32_t>& gridsId = pixel.second;
					std::uint32_t maxColorScoreGrid = gridsId[0];
					float maxColorScoreGridColorScore = gridAccumScore[maxColorScoreGrid];
					for (size_t i = 1; i < gridsId.size(); i++)
					{
						const std::uint32_t& thisGridId = gridsId[i];
						const float& thisGridColorScore = gridAccumScore[thisGridId];
						if (thisGridColorScore > maxColorScoreGridColorScore)
						{
							maxColorScoreGridColorScore = thisGridColorScore;
							maxColorScoreGrid = thisGridId;
						}
					}
					PosEncodeToGridXyz(maxColorScoreGrid, this->resolutionX, this->resolutionXY, gridX, gridY, gridZ);
					ImgPixelEncodeToImgXy(pixelId, x_2d, y_2d, mask.cols);
					float x = (0.5 + gridX) * this->gridResolution + this->targetMinBorderX;
					float y = (0.5 + gridY) * this->gridResolution + this->targetMinBorderY;
					float z = (0.5 + gridZ) * this->gridResolution + this->targetMinBorderZ;
					//fout1 << x << " " << y << " " << z << " " << maxColorScoreGridColorScore << std::endl;
					ptsMat.at<cv::Vec3f>(y_2d, x_2d)[0] = x;
					ptsMat.at<cv::Vec3f>(y_2d, x_2d)[1] = y;
					ptsMat.at<cv::Vec3f>(y_2d, x_2d)[2] = z;
					ptsMatMask.ptr<uchar>(y_2d)[x_2d] = 1;
				}
				//fout1.close();

				cv::FileStorage fs("../surf/test" + std::to_string(thisView.viewId) + ".yml", cv::FileStorage::WRITE);
				fs << "ptsMatMask" << ptsMatMask;
				fs << "ptsMat" << ptsMat;
				fs.release(); 

			}


			saveGrid(outDir /"grid.g", gridAccumScore);


			return 0;
		}
		 
		static float getColorScore(const std::vector<std::uint8_t>& inWhichViews, const std::vector<cv::Vec3b>& rgbSet)
		{
			std::vector<float>redSet;
			std::vector<float>greenSet;
			std::vector<float>blueSet; 
			redSet.reserve(inWhichViews.size());
			greenSet.reserve(inWhichViews.size());
			blueSet.reserve(inWhichViews.size());
			int cnt = 0;
			for (int i = 0; i < inWhichViews.size(); i++)
			{
				if (inWhichViews[i]>0)
				{
					cnt += 1;
					redSet.emplace_back(rgbSet[i][0]);
					greenSet.emplace_back(rgbSet[i][1]);
					blueSet.emplace_back(rgbSet[i][2]);
				}
			}
			if (cnt<=3)
			{
				return 0;
			}
			std::pair<float, float>redMeanVar = calVarStdev(redSet);
			std::pair<float, float>greenMeanVar = calVarStdev(greenSet);
			std::pair<float, float>blueMeanVar = calVarStdev(blueSet);

			return redMeanVar.second + greenMeanVar.second + blueMeanVar.second ;
		}
	private:
		std::filesystem::path dataDir;
		std::filesystem::path objPath;
		Eigen::MatrixXf vertex;
		Eigen::MatrixXi faces;
		float targetMinBorderX;
		float targetMinBorderY;
		float targetMinBorderZ;
		float targetMaxBorderX;
		float targetMaxBorderY;
		float targetMaxBorderZ;		
		int resolutionX;
		int resolutionY;
		int resolutionZ;
		int resolutionXY;
		float gridResolution;
		float gridResolutionInv;
		GridConfig gridConfig;
		std::unordered_map<int, Camera>cameras;
		std::unordered_map<int, View>views;
		std::unordered_map<std::uint32_t, std::vector<std::uint8_t>>gridInViews;
		std::unordered_map<std::uint32_t, std::vector<cv::Vec3b>>gridRgbset;
	};

}

int test_mrf()
{
	cv::Mat ptsMatMask, ptsMat;
	cv::FileStorage fs( "../surf/test0.yml",cv::FileStorage::READ);
	fs["ptsMatMask"] >> ptsMatMask;
	fs["ptsMat"] >> ptsMat;
	fs.release();
	std::vector<cv::Point3f> pts;
	std::vector<cv::Point3i> faces;
	mrf::generThinMesh(ptsMatMask, ptsMat, pts,faces);
	LOG_OUT; 
	return 0;


	float gridResolution = 0.03;//need measured before
	mrf::GridConfig gridConfig = { 8,gridResolution };
	//mrf::Mrf asd("../data/a/result", "../data/a/result/dense.obj", gridConfig);
	//asd.serialization("../surf/d.dat");
	mrf::Mrf asd2;
	asd2.reload("../surf/d.dat");
	asd2.mrf1();

	//cv::Mat ptsMatMask, ptsMat;
	//cv::FileStorage fs("../surf/test0.yml", cv::FileStorage::READ);
	//fs["ptsMatMask"] >> ptsMatMask;
	//fs["ptsMat"] >> ptsMat;
	//fs.release();
	//std::vector<cv::Point3f> pts;
	//std::vector<cv::Point3i> faces;
	//mrf::generThinMesh(ptsMatMask, ptsMat, pts, faces);
	//LOG_OUT;

	return 0;
}