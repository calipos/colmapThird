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

//std::map<std::filesystem::path, std::vector<std::string>> 
std::map<Camera, std::vector<Image>> loadImageData(const std::filesystem::path& dir, const ImageIntrType&type);
bool convertDataset(const std::map<Camera, std::vector<Image>>& d, std::vector<Camera>& cameraList, std::vector<Image>& imageList);


#endif // !_SCENE_H_
