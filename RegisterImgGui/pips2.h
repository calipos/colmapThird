#ifndef _PIPS2_H_
#define _PIPS2_H_
#include <filesystem>
#include <vector>
#include "opencv2/opencv.hpp"
#include "mat.h"
#include "net.h"
namespace pips2
{
	class Pips2
	{
	public:
		Pips2(
			const std::filesystem::path& ncnnEncoderParamPath, const std::filesystem::path& ncnnEncoderBinPath);
		~Pips2();
		bool inputImage(const std::vector<std::string>& imgPath, std::vector<ncnn::Mat>& fmap);
		bool inputImage(const cv::Mat& img,ncnn::Mat& fmap);
		//bool serializationFeat(const std::filesystem::path& path);
		//bool deserializationFeat(const std::filesystem::path& path);
		cv::Size imgSize;
	private:
		bool changeParamResizeParam(const std::string& path, const std::pair<int, int>& d);
		ncnn::Net encoderNet;
		std::filesystem::path ncnnEncoderParamPath_, ncnnEncoderBinPath_;
	};
}
int test_pips2();

#endif // !_PIPS2_H_
