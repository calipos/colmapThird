#pragma once
#include <filesystem>
#include "Eigen/Core"
#include "opencv2/opencv.hpp"
namespace meshdraw
{
	template <typename T>
	bool isEmpty(const Eigen::MatrixBase<T>& mat) {
		if (mat.rows() == 0 || mat.cols() == 0) {
			return true;
		}
		return false;
	}
	enum class CmaeraType
	{
		Pinhole=1,
		Ortho,
	};
	struct Camera
	{
		CmaeraType cameraType{ CmaeraType::Pinhole };
		Eigen::Matrix3f intr;
		Eigen::Matrix3f R;
		Eigen::RowVector3f t;
		int height;
		int width;
	};
	struct Mesh
	{
		Mesh();
		Mesh(const Eigen::MatrixX3f V_, const Eigen::MatrixX3i& F_, const Eigen::MatrixX3f& C_);
		Eigen::MatrixX3f V;
		Eigen::MatrixX3i F;
		Eigen::MatrixX3f C;
		Eigen::MatrixX3f ptsNormal;
		Eigen::MatrixX3f facesNormal;
		bool figurePtsNomral();
		bool figureFacesNomral();
		bool rotate(const Eigen::Matrix3f& R, const Eigen::RowVector3f& t);
	};
	enum class RenderType
	{
		vertexColor= 1,
		distance,
	};
	bool render(const Mesh&msh,const Camera&cam, cv::Mat& rgbMat, cv::Mat& vertexMap, cv::Mat& mask, const RenderType&renderTpye = RenderType::vertexColor);

	namespace utils
	{
		meshdraw::Camera generateBfmDefaultCamera();
		meshdraw::Camera generateDefaultCamera();
		Eigen::Matrix3f generRotateMatrix(const Eigen::Vector3f& direct, const Eigen::Vector3f& upDirect);
		bool saveFacePickedMesh(const std::filesystem::path&path,const Mesh&msh,const std::vector<bool>&faceValid);
		bool savePtsMat(const std::filesystem::path& path, const cv::Mat& ptsMat, const cv::Mat& mask);
	}

}