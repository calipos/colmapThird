#ifndef _SAM2_H_
#define _SAM2_H_
#include <vector>
#include <filesystem>
#include <vector>
#include <optional>
#include "net.h"
#include "opencv2/opencv.hpp"
namespace sam2
{
	class Sam2
	{
	public:
		Sam2(
			const std::filesystem::path& ncnnEncoderParamPath, const std::filesystem::path& ncnnEncoderBinPath,
			const std::filesystem::path& onnxDecoderPath);
		~Sam2();
		bool inputImage(const std::filesystem::path& imgPath);
		bool inputImage(const cv::Mat& img);
		bool inputHint();
		bool inputHint(const std::vector<std::pair<int, cv::Point2i>>& hint, cv::Mat& mask);
		bool inputSingleHint(const float& hintx, const float& hinty, const int labelId, const cv::Mat& inPutMask, cv::Mat& mask);
		bool serializationFeat(const std::filesystem::path& path);
		bool deserializationFeat(const std::filesystem::path& path);
		const static std::vector<int>high_res_feats_0_shape;
		const static std::vector<int>high_res_feats_1_shape;
		const static std::vector<int>image_embed_shape;
	public:
		ncnn::Net encoderNet;
		std::optional < cv::dnn::Net> positionDecoderNet;
		cv::Mat high_res_feats_0;
		cv::Mat high_res_feats_1;
		cv::Mat image_embed;
		cv::Size oringalSize;
	};

}
#endif // !_SAM2_H_
