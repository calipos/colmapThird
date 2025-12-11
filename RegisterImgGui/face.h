#pragma once
#include "opencv2/opencv.hpp"
#include "opencv2/face.hpp"
namespace face
{
	class FaceDet
	{
	public:
		FaceDet();
		~FaceDet();
		bool init();
		cv::Ptr<cv::FaceDetectorYN> faceDetector{nullptr};
		bool detect(const cv::Mat& img, std::vector<cv::Rect>& rects, std::vector<std::vector<cv::Point2f>>& faceLandmarks, std::vector<float>& scores);
		const int inputWidth{ 640 };
		const int inputHeight{ 640 };
	private:
		cv::Mat resizeImg640(const cv::Mat& image, double& factor);
	};
	class FaceMark
	{
	public:
		FaceMark();
		~FaceMark();
		bool init();
		cv::Ptr<cv::face::Facemark> faceFaceMarker{ nullptr };
		bool extract(const cv::Mat& img, const std::vector<cv::Rect>& faceRect, std::vector<std::vector<cv::Point2f> >& landmarks);
		
	};
}