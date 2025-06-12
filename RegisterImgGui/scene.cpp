#include <vector>
#include "image.h"
#include "scene.h"
#include "labelme.h"
#include "camera.h"
std::map<Camera, std::vector<Image>> loadImageData(const std::filesystem::path& dir, const ImageIntrType& type)
{
	std::map<std::filesystem::path, std::map<std::string, Eigen::Vector2d>>corners;
	std::map<std::filesystem::path, Eigen::Vector2i>imgSizeWHs;
	std::map<std::filesystem::path, camera_t>dirFlags;
	camera_t dirFlag = 0;
	for (auto const& dir_entry : std::filesystem::recursive_directory_iterator{ dir })
	{
		const auto& thisFilename = dir_entry.path();
		if (thisFilename.has_extension())
		{
			const auto& shortName = thisFilename.filename().stem().string();
			const auto& ext = thisFilename.extension().string();
			if (ext.compare(".json") == 0)
			{
				std::map<std::string, Eigen::Vector2d> cornerInfo;
				Eigen::Vector2i imgSizeWH;
				std::string imgPath;
				bool readret = labelme::readPtsFromLabelMeJson(thisFilename, cornerInfo, imgSizeWH, &imgPath);
				if (readret &&std::filesystem::exists(thisFilename.parent_path()/imgPath))
				{
					corners[thisFilename] = cornerInfo;
					imgSizeWHs[thisFilename] = imgSizeWH;
					auto parentDir = thisFilename.parent_path();
					if (dirFlags.count(parentDir)==0)
					{
						dirFlags[parentDir] = dirFlag++;
					}
				}
			}
		}
	}

	Image::keypointNameToIndx.clear();;
	Image::keypointIndexToName.clear();;
	std::map<Camera, std::vector<Image>>dataSet;
	const auto& defaultCameraType = CameraModelId::kSimpleRadial;
	LOG_OUT << "use default focus length = 1.2*max(w,h)";
	LOG_OUT << "use default CameraType = "<<CameraModelIdToName(defaultCameraType);
	camera_t camera_id = 0;
	image_t image_id = 0;
	for (const auto&d: corners)
	{
		const auto& path = d.first;
		const auto& cornerInfo = d.second;
		const auto& sizeWH = imgSizeWHs[path];
		Image thisImg;
		thisImg.SetImageId(image_id++);
		std::string imgName = path.filename().stem().string();
		std::string imgDirName = path.parent_path().filename().stem().string();
		imgName = imgDirName + '@' + imgName;
		thisImg.SetName(imgName);
		for (const auto&feat: cornerInfo)
		{
			const auto& featName = feat.first;
			const auto& featPos = feat.second;
			if (Image::keypointNameToIndx.count(featName)==0)
			{
				int featId_ = Image::keypointNameToIndx.size();
				Image::keypointNameToIndx[featName] = featId_;
				Image::keypointIndexToName[featId_] = featName;
			}
			const int& featId = Image::keypointNameToIndx[featName];
			thisImg.featPts[featId] = featPos;
		}
		thisImg.SetPoints2D(thisImg.featPts);
		if (type== ImageIntrType::DIFFERENT)
		{
			double focal_length = 1.2 * std::max(sizeWH.x(), sizeWH.y());
			Camera temp = Camera::CreateFromModelId(camera_id++, defaultCameraType, focal_length, sizeWH.x(), sizeWH.y());
			bool has_focal_length = false;
			temp.has_prior_focal_length = has_focal_length;
			thisImg.SetCameraId(temp.camera_id);
			dataSet[temp].emplace_back(thisImg);
		}
		else if (type == ImageIntrType::SHARED_WITH_FOLDER)
		{
			double focal_length = 1.2 * std::max(sizeWH.x(), sizeWH.y());
			camera_id = dirFlags[path.parent_path()];
			Camera temp = Camera::CreateFromModelId(camera_id, defaultCameraType, focal_length, sizeWH.x(), sizeWH.y());
			bool has_focal_length = false;
			temp.has_prior_focal_length = has_focal_length;
			thisImg.SetCameraId(temp.camera_id);
			dataSet[temp].emplace_back(thisImg);
			
		}
		else if (type == ImageIntrType::SHARED_ALL)
		{
			double focal_length = 1.2 * std::max(sizeWH.x(), sizeWH.y());
			Camera temp = Camera::CreateFromModelId(0, defaultCameraType, focal_length, sizeWH.x(), sizeWH.y());
			bool has_focal_length = false;
			temp.has_prior_focal_length = has_focal_length;
			thisImg.SetCameraId(temp.camera_id);
			dataSet[temp].emplace_back(thisImg);
		}		
	}

	return  dataSet;
}
bool convertDataset(const std::map<Camera, std::vector<Image>>& d, std::vector<Camera>&cameraList, std::vector<Image>& imageList)
{
	for (const auto&d1:d)
	{
		cameraList.emplace_back(d1.first);
		for (const auto&d2:d1.second)
		{
			imageList.emplace_back(d2);
		}
	}
	return true;
}