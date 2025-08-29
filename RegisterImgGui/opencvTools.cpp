#include <filesystem>
#include <fstream>
#include "opencvTools.h"
#include "log.h"
namespace tools
{
	bool saveMask(const std::string& path, const cv::Mat& mask)
	{
		if (mask.empty())
		{
			LOG_ERR_OUT << "empty.";
			return false;
		}
		if (mask.channels()!=1)
		{
			LOG_ERR_OUT << "mask.channels()!=1";
			return false;
		}
		std::fstream fout(path, std::ios::out | std::ios::binary);
		int height = mask.rows;
		int width = mask.cols;
		int total = mask.dataend - mask.data;
		fout.write((char*)&height, sizeof(int));
		fout.write((char*)&width, sizeof(int));
		fout.write((char*)&total, sizeof(int));
		fout.write((char*)mask.data, total*sizeof(uchar));
		fout.close();
		return true;
	}
	cv::Mat loadMask(const std::string& path)
	{
		if (!std::filesystem::exists(path))
		{
			LOG_ERR_OUT << "not found : " << path;
			return cv::Mat();
		}
		try
		{
			std::fstream fin(path, std::ios::in | std::ios::binary);
			int height = 0;
			int width = 0;
			int total = 0;
			fin.read((char*)&height, sizeof(int));
			fin.read((char*)&width, sizeof(int));
			fin.read((char*)&total, sizeof(int));
			cv::Mat mask(height, width, CV_8UC1);
			if (mask.dataend - mask.data!= total)
			{
				LOG_ERR_OUT << "mask.dataend - mask.data!= total";
				return cv::Mat();
			}
			fin.read((char*)mask.data, total * sizeof(uchar));
			fin.close();
			return mask;
		}
		catch (const std::exception&)
		{
			LOG_ERR_OUT << "catch from : " << path;
			return cv::Mat();
		}
	}
}