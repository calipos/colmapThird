#include <filesystem>
#include "face.h"
#include "log.h"
namespace face
{ 
	FaceDet::FaceDet()
	{ 
	}
	bool FaceDet::init()
	{
		if (faceDetector!=nullptr)
		{
			LOG_OUT << "already initialized.";
			return true;
		}
		const std::filesystem::path faceDetModelPath = "../models/face_detection_yunet_2023mar.onnx";
		if (!std::filesystem::exists(faceDetModelPath))
		{
			LOG_ERR_OUT << "not found : "<< faceDetModelPath;
			return false;
		}
		try
		{
			faceDetector = cv::FaceDetectorYN::create("../models/face_detection_yunet_2023mar.onnx", "", cv::Size(640, 640));
		}
		catch (...)
		{
			LOG_ERR_OUT << "load model fail. : " << faceDetModelPath;
			return false;
		}
		if (faceDetector == nullptr)
		{
			LOG_OUT << "init fail.";
			return false;
		}
		return true;
	}
	FaceDet::~FaceDet()
	{
		faceDetector = nullptr;
	}
	bool FaceDet::detect(const cv::Mat& img, std::vector<cv::Rect>& rects, std::vector<std::vector<cv::Point2f>>& faceLandmarks, std::vector<float>& scores)
	{
		if (img.empty())
		{
			LOG_ERR_OUT << "img empty";
			return false;
		}
		if (faceDetector==nullptr)
		{
			LOG_ERR_OUT << "init first";
			return false;
		}
		double resizeFactor = 1.;
		cv::Mat imgOri;
		if (img.size()!=cv::Size(640,640))
		{
			imgOri = resizeImg640(img, resizeFactor);
		}
		else
		{
			img.copyTo(imgOri);
		}
		if (imgOri.channels()==1)
		{
			cv::cvtColor(imgOri, imgOri,cv::COLOR_GRAY2RGB);
		}
		double resizeFactorInv = 1. / resizeFactor;
		cv::Mat facesMat;
		faceDetector->detect(imgOri, facesMat);
		rects.resize(facesMat.rows);
		faceLandmarks.resize(facesMat.rows, std::vector<cv::Point2f>(5));
		scores.resize(facesMat.rows);
		for (int i = 0; i < facesMat.rows; i++)
		{
			rects[i].x = facesMat.at<float>(i, 0) * resizeFactorInv;
			rects[i].y = facesMat.at<float>(i, 1) * resizeFactorInv;
			rects[i].width = facesMat.at<float>(i, 2) * resizeFactorInv;
			rects[i].height = facesMat.at<float>(i, 3) * resizeFactorInv;

			faceLandmarks[i][0].x = facesMat.at<float>(i, 4)* resizeFactorInv;; faceLandmarks[i][0].y = facesMat.at<float>(i, 5) * resizeFactorInv;
			faceLandmarks[i][1].x = facesMat.at<float>(i, 6)* resizeFactorInv;; faceLandmarks[i][1].y = facesMat.at<float>(i, 7) * resizeFactorInv;
			faceLandmarks[i][2].x = facesMat.at<float>(i, 8)* resizeFactorInv;; faceLandmarks[i][2].y = facesMat.at<float>(i, 9) * resizeFactorInv;
			faceLandmarks[i][3].x = facesMat.at<float>(i, 10)* resizeFactorInv;; faceLandmarks[i][3].y = facesMat.at<float>(i, 11) * resizeFactorInv;
			faceLandmarks[i][4].x = facesMat.at<float>(i, 12)* resizeFactorInv;; faceLandmarks[i][4].y = facesMat.at<float>(i, 13) * resizeFactorInv;

			scores[i] = facesMat.at<float>(i, 14);
		}
		return true;
	}
	cv::Mat FaceDet::resizeImg640(const cv::Mat& image, double& factor)
	{
		int max = (std::max)(image.rows, image.cols);
		factor = 640. / max;
		cv::Mat img;
		cv::resize(image, img, cv::Size(), factor, factor);
		cv::Mat ret = cv::Mat::zeros(640, 640, img.type());
		img.copyTo(ret(cv::Rect(0, 0, img.cols, img.rows)));
		return ret;
	}




	FaceMark::FaceMark() {}
	FaceMark::~FaceMark() { faceFaceMarker = nullptr; }
	bool FaceMark::init()
	{
		if (faceFaceMarker != nullptr)
		{
			LOG_OUT << "already initialized.";
			return true;
		}
		const std::filesystem::path faceMarkModelPath = "../models/lbfmodel.yaml";
		if (!std::filesystem::exists(faceMarkModelPath))
		{
			LOG_ERR_OUT << "not found : " << faceMarkModelPath;
			return false;
		}
		try
		{
			faceFaceMarker = cv::face::FacemarkLBF::create();
			faceFaceMarker->loadModel("../models/lbfmodel.yaml");
		}
		catch (...)
		{
			LOG_ERR_OUT << "load model fail. : " << faceMarkModelPath;
			return false;
		}
		if (faceFaceMarker == nullptr)
		{
			LOG_OUT << "init fail.";
			return false;
		}
		return true;
	} 
	bool FaceMark::extract(const cv::Mat& img,const std::vector<cv::Rect>&faceRect, std::vector<std::vector<cv::Point2f> >& landmarks)
	{
		if (img.empty())
		{
			LOG_ERR_OUT << "img empty";
			return false;
		}
		landmarks.clear();
		return faceFaceMarker->fit(img, faceRect, landmarks);
	}

}