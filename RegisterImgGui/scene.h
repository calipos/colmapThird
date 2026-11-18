#ifndef _SCENE_H_
#define _SCENE_H_
#include <map>
#include <filesystem>
#include <vector>
#include <string>
#include "camera.h"
#include "image.h"
#include "eigen_alignment.h"


enum class ImageIntrType
{
	SHARED_WITH_FOLDER = 1,
	SHARED_ALL = 2,
	DIFFERENT = 3, 
};
struct SceneData
{
	float x;
	float y;
	float z;
	std::uint8_t r;
	std::uint8_t g;
	std::uint8_t b;
	std::uint32_t i; 
	SceneData(float x_, float y_, float z_,
		std::uint8_t r_, std::uint8_t g_,
		std::uint8_t b_, std::uint32_t i_)
		: x(x_), y(y_), z(z_), r(r_), g(g_), b(b_), i(i_) {}
};
//std::map<std::filesystem::path, std::vector<std::string>> 
std::map<Camera, std::vector<Image>> loadImageData(const std::filesystem::path& dir, const ImageIntrType&type);
bool convertDataset(std::map<Camera, std::vector<Image>>& d, std::vector<Camera>& cameraList, std::vector<Image>& imageList);
bool writeResult(const std::filesystem::path& resultDir,
	const std::vector<Camera>& cameraList,
	const std::vector<Image>& imageList,
	const std::unordered_map<point3D_t, Eigen::Vector3d>& objPts,
	const std::unordered_map < image_t, struct Rigid3d>& poses);


#endif // !_SCENE_H_
