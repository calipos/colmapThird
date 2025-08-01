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
			const std::filesystem::path& ncnnEncoderParamPath, const std::filesystem::path& ncnnEncoderBinPath,const int&radius_=3);
		~Pips2();
		bool inputImage(const std::vector<std::string>& imgPath, std::vector<ncnn::Mat>& fmaps);
		bool inputImage(const cv::Mat& img,ncnn::Mat& fmap);
		//bool serializationFeat(const std::filesystem::path& path);
		//bool deserializationFeat(const std::filesystem::path& path);
		cv::Size imgSize;
		std::vector<cv::Size>fmapSize;
		int radius;
		std::vector<float>coord_delta_x;
		std::vector<float>coord_delta_y;
		static std::string getBilinearOpNet();
		static std::string getCorrsNet(const int& sequenceLength, const int& imgHeight, const int& imgWidth);
		static std::string getDeltaInNer(const int& sequenceLength);
		static ncnn::Mat bilinear_sample2d(const ncnn::Mat& blob, const std::vector<float>& xs, const std::vector<float>& ys, std::shared_ptr<ncnn::Net> bilinearOpNet);
		static ncnn::Mat bilinear_sample2d(const ncnn::Mat& blob, const std::vector<std::vector<float>>& xs, const std::vector<std::vector<float>>& ys, std::shared_ptr<ncnn::Net> bilinearOpNet, const int& padding_mode = 0);
		static ncnn::Mat concatFmaps(const std::vector<ncnn::Mat>&fmap, const std::vector<int>& picks);
		static ncnn::Mat repeatFeat(const ncnn::Mat&feat, const int&s);
		static std::vector<std::vector<float>> expandInitCoord(std::vector<float>& xs, const int& times);
		bool fillPositionDiffCosSin(const ncnn::Mat& corr1, const ncnn::Mat& corr2, const ncnn::Mat& corr4, const std::vector<std::vector<float>>& stride_x, const std::vector<std::vector<float>>& stride_y);
		ncnn::Mat pyramidSample(const std::vector<ncnn::Mat>& corrs_pyramids, const std::vector<std::vector<float>>& stride_x, const std::vector<std::vector<float>>& stride_y)const;
		std::shared_ptr<ncnn::Net> bilinearOpNet = 0;
		std::shared_ptr<ncnn::Net> corrsNet = 0;
		std::shared_ptr<ncnn::Net> positionDiffEncoderNet = 0;
		static const int stride;
		static const int corrsBlockCnt;
		static const int latent_dim;
		static const int pyramid_level;
		static const int omega_temperature;
	private:
		ncnn::Mat omega;		
		bool changeParamResizeParam(const std::string& path, const std::pair<int, int>& d);
		ncnn::Net encoderNet;
		std::filesystem::path ncnnEncoderParamPath_, ncnnEncoderBinPath_;
	};
}
int test_pips2();

#endif // !_PIPS2_H_
