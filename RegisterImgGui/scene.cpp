#include <fstream>
#include <unordered_map>
#include <vector>
#include "image.h"
#include "scene.h"
#include "labelme.h"
#include "camera.h"
#include "json/json.h"
#include "bitmap.h"
#include "undistortion.h"
#include "misc.h"
#include "warp.h"
std::map<Camera, std::vector<Image>> loadImageData(const std::filesystem::path& dir, const ImageIntrType& type)
{
	std::map<std::filesystem::path, std::map<std::string, Eigen::Vector2d>>corners;
	std::map<std::filesystem::path, std::filesystem::path>picPath;
	std::map<std::filesystem::path, Eigen::Vector2i>imgSizeWHs;
	std::map<std::filesystem::path, camera_t>dirFlags;
	camera_t dirFlag = 0;
	for (auto const& dir_entry : std::filesystem::recursive_directory_iterator{ dir })
	{
		const auto& thisFilename = dir_entry.path();
		auto parentName = thisFilename.parent_path().filename().string();
		if (parentName.compare("result") == 0)
		{
			continue;
		}
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
					picPath[thisFilename] = thisFilename.parent_path() / imgPath;
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
	const auto& defaultCameraType = CameraModelId::kSimplePinhole;
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
		//imgName = imgDirName + '@' + imgName;
		thisImg.SetName(picPath[path].string());
		for (const auto&feat: cornerInfo)
		{
			const auto& featName = feat.first;
			const auto& featPos = feat.second;
			if (Image::keypointNameToIndx.count(featName)==0)
			{
				point2D_t featId_ = Image::keypointNameToIndx.size();
				Image::keypointNameToIndx[featName] = featId_;
				Image::keypointIndexToName[featId_] = featName;
			}
			const int& featId = Image::keypointNameToIndx[featName];
			thisImg.featPts[featId] = featPos;
		}
		if (!thisImg.SetPoints2D(thisImg.featPts))
		{
			LOG_ERR_OUT << "feat formate error. must be [0: 1: 2: ...]";
		}
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
bool convertDataset(std::map<Camera, std::vector<Image>>& d, std::vector<Camera>&cameraList, std::vector<Image>& imageList)
{
	for (auto&d1:d)
	{
		const Camera& camera = d1.first;
		cameraList.emplace_back(camera);
		for (auto&d2:d1.second)
		{
			d2.SetCameraPtr(&cameraList.back());
			imageList.emplace_back(d2);
		}
	}
	return true;
}
bool writeResult(const std::filesystem::path& dataDir,
	const std::vector<Camera>& cameraList,
	const std::vector<Image>& imageList,
	const std::unordered_map<point3D_t, Eigen::Vector3d>& objPts,
	const std::unordered_map < image_t, struct Rigid3d>& poses)
{
	if (std::filesystem::exists(dataDir))
	{
		removeDirRecursive(dataDir);
	}
	std::filesystem::create_directories(dataDir); 

	UndistortCameraOptions undistortion_options;
	std::unordered_map<camera_t, Camera>undistortedCameraMap;
	for (const auto&[imgId,Rt]: poses)
	{
		const Image& image1 = imageList[imgId];
		camera_t cameraId = image1.CameraId();
		std::filesystem::path originalPath(image1.Name());
		auto subDirName = originalPath.parent_path().filename().string();
		auto fileName = originalPath.filename().stem().string();
		auto newImgPath = dataDir / (subDirName + "@" + fileName + ".jpg");
		auto imgJsonPath = dataDir / (subDirName + "@" + fileName + ".json");
		if (undistortedCameraMap.count(cameraId)==0)
		{
			const Camera& camera1 = cameraList[cameraId];
			undistortedCameraMap[cameraId] = UndistortCamera(undistortion_options, camera1);
		}

		const Camera& distorted_camera = cameraList[cameraId];
		const Camera& undistorted_camera = undistortedCameraMap[cameraId];

		Bitmap distorted_bitmap;
		distorted_bitmap.Read(image1.Name());
		Bitmap  undistorted_bitmap;//
		WarpImageBetweenCameras(distorted_camera,
			undistorted_camera,
			distorted_bitmap,
			&undistorted_bitmap);
		distorted_bitmap.CloneMetadata(&undistorted_bitmap);
		undistorted_bitmap.Write(newImgPath.string());


		newImgPath = std::filesystem::canonical(newImgPath);
		Json::Value labelRoot;
		labelRoot["version"] = Json::Value("1");
		labelRoot["imagePath"] = Json::Value(newImgPath.string());
		labelRoot["fx"] = Json::Value(undistorted_camera.params[0]);
		labelRoot["fy"] = Json::Value(undistorted_camera.params[1]);
		labelRoot["cx"] = Json::Value(undistorted_camera.params[2]);
		labelRoot["cy"] = Json::Value(undistorted_camera.params[3]);
		labelRoot["width"] = Json::Value(undistorted_camera.width);
		labelRoot["height"] = Json::Value(undistorted_camera.height);
		Json::Value Qt;
		Qt.append(Rt.rotation.w());
		Qt.append(Rt.rotation.x());
		Qt.append(Rt.rotation.y());
		Qt.append(Rt.rotation.z());
		Qt.append(Rt.translation.x());
		Qt.append(Rt.translation.y());
		Qt.append(Rt.translation.z());
		labelRoot["Qt"] = Qt;
		Json::StyledWriter sw;
		std::fstream fout(imgJsonPath, std::ios::out);
		fout << sw.write(labelRoot);
		fout.close();



		Eigen::MatrixXd pts3d(4, objPts.size());
		Eigen::MatrixXd pts2d(2, objPts.size());
		int idx = 0;
		for (const auto& [ptId, pt] : objPts)
		{
			pts3d(0, idx) = pt.x();
			pts3d(1, idx) = pt.y();
			pts3d(2, idx) = pt.z();
			pts3d(3, idx) = 1.;
			idx++;
		}
		LOG_OUT << newImgPath;
		LOG_OUT << pts3d;
		Rigid3d  Rtinv = Inverse(Rt);
		Eigen::MatrixXd pts3d2 = Rtinv.ToMatrix() * pts3d;
		pts3d2 = Rt.ToMatrix() * pts3d;
		LOG_OUT << undistorted_camera.params[0] << " " << undistorted_camera.params[1] << " " << undistorted_camera.params[2] << " " << undistorted_camera.params[3] << " ";
		for (int i = 0; i < objPts.size(); i++)
		{
			pts2d(0, i) = pts3d2(0, i) / pts3d2(2, i) * undistorted_camera.params[0] + undistorted_camera.params[2];
			pts2d(1, i) = pts3d2(1, i) / pts3d2(2, i) * undistorted_camera.params[1] + undistorted_camera.params[3];
		}
		LOG_OUT << pts2d;
		LOG_OUT<<"=======================================================";
	}

	std::fstream fPtsTxt(dataDir / "pts.txt", std::ios::out);
	Json::Value labelRoot;	 
	std::filesystem::path ptsJsonPath = dataDir / "pts.json";
	for (const auto&[ptId,pt]: objPts)
	{
		Json::Value ptNode;
		const std::string& ptName = Image::keypointIndexToName.at(ptId);
		ptNode["idx"] = Json::Value(ptId);
		ptNode["name"] = Json::Value(ptName);
		ptNode["xyz"] = Json::Value();
		ptNode["xyz"].append(pt.x());
		ptNode["xyz"].append(pt.y());
		ptNode["xyz"].append(pt.z());
		labelRoot.append(ptNode);
		fPtsTxt << pt.x() << " " << pt.y() << " " << pt.z() << std::endl;
	}
	Json::StyledWriter sw;
	std::fstream fout(ptsJsonPath.string(), std::ios::out);
	fout << sw.write(labelRoot);
	fout.close();
	fPtsTxt.close();
	return true;
}