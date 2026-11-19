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
#include "bitmap.h"
#include "opencv2/opencv.hpp"
std::map<Camera, std::vector<Image>> loadImageData(const std::filesystem::path& dir, const ImageIntrType& type)
{
	std::map<std::filesystem::path, std::map<std::string, Eigen::Vector2d>>corners;
	std::map<std::filesystem::path, std::filesystem::path>picPath;
	std::map<std::filesystem::path, std::filesystem::path>jsonPath;
	std::map<std::filesystem::path, Eigen::Vector2i>imgSizeWHs;
	std::map<std::filesystem::path, camera_t>dirFlags;
	camera_t dirFlag = 0;
	for (auto const& dir_entry : std::filesystem::recursive_directory_iterator{ dir })
	{
		const auto& thisFilename = dir_entry.path();
		auto parentName = thisFilename.parent_path().filename().string();
		if (parentName.find("result") == 0)
		{
			continue;
		}
		if (parentName.compare("temp") == 0)
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
					jsonPath[thisFilename] = thisFilename;
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
	Image::picNameToIndx.clear();
	Image::picIndexTopicName.clear();
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
		

		std::string imgName = path.filename().stem().string();
		std::string imgDirName = path.parent_path().filename().stem().string();
		//imgName = imgDirName + '@' + imgName;
		Image::picNameToIndx[imgName] = image_id;
		Image::picIndexTopicName[image_id] = imgName;
		thisImg.SetImageId(image_id++);
		thisImg.SetName(imgName);
		thisImg.SetPicPath(picPath[path].string());
		thisImg.SetJsonPath(jsonPath[path].string());
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
std::vector<SceneData> getCameraCloudPoint(const struct Rigid3d& pose, const int& height, const int& width,const float& cameraScale,const float&cameraId)
{
	auto randColor = BitmapColor<std::uint8_t>::randColor();
	Eigen::Quaternionf q(pose.rotation.w(), pose.rotation.x(), pose.rotation.y(), pose.rotation.z());
	Eigen::Matrix3f R = q.matrix();
	Eigen::Vector3f t(pose.translation.x(), pose.translation.y(), pose.translation.z());
	int cameraBack = sqrt(height * height + width * width)*2;
	int interval = 100;
	std::vector<SceneData> cloud;
	cloud.reserve(1024);
	int edge = (std::max)(height, width);
	float cameraScale2 = cameraScale / edge;
	for (int z = 0; z >= -cameraBack; z -= interval)
	{
		float scale = 1.f*(z + cameraBack) / cameraBack; 
		int Y = (height >> 1) * scale;
		int X = (width >> 1) * scale;
		int endY = 2 * Y / interval * interval - Y;
		int endX = 2 * X / interval * interval - X;
		for (int y = -Y; y <= endY; y += interval)
		{
			for (int x = -X; x <= endX; x += interval)
			{
				if (y != -Y && y != endY && x != -X && x != endX)
				{
					continue;
				}
				Eigen::Vector3f p= t + R * Eigen::Vector3f(x,y,z) * cameraScale2;
				cloud.emplace_back(p[0], p[1], p[2], randColor.r, randColor.g, randColor.b, cameraId);
			}
		}
	}
	return cloud;
}
std::vector<SceneData> getCameraTrajectories(const std::vector<Eigen::Vector3d>&cameraTs)
{
	std::vector<SceneData> ret;
	if (cameraTs.size()<2)
	{
		return ret;
	}
	std::vector<float>dists(cameraTs.size() - 1);
	std::vector<float>accDists(cameraTs.size(),0);
	for (size_t i = 0; i < cameraTs.size() - 1; i++)
	{
		dists[i] = (cameraTs[i] - cameraTs[i + 1]).norm();
		accDists[i+1] = accDists[i] + dists[i];
	}
	float min_dist = *std::min_element(dists.begin(), dists.end())+1e-9;
	float step = min_dist / 20;
	int totalStep = std::ceil(accDists.back() / step);
	LOG_OUT << "totalStep = "<< totalStep;
	ret.reserve(totalStep);
	for (int i = 0; i < totalStep; i++)
	{
		float currentPos = step * i;
		float currentPosRatio = currentPos / accDists.back();
		if (currentPos> accDists.back())
		{
			break;
		}
		std::uint8_t r = JetColormap::Red(currentPosRatio) * 255;
		std::uint8_t g = JetColormap::Green(currentPosRatio) * 255;
		std::uint8_t b = JetColormap::Blue(currentPosRatio) * 255;
		float x = 0, y = 0, z = 0;
		for (size_t j = 0; j < accDists.size()-1; j++)
		{
			if (accDists[j]<= currentPos && (accDists[j+1]+1e-3)>= currentPos)
			{
				double beta = (currentPos - accDists[j]) / dists[j];
				double alpha = 1 - beta;
				x = alpha * cameraTs[j].x() + beta * cameraTs[j + 1].x();
				y = alpha * cameraTs[j].y() + beta * cameraTs[j + 1].y();
				z = alpha * cameraTs[j].z() + beta * cameraTs[j + 1].z();
				ret.emplace_back(x, y, z, r, g, b, 254);
				break;
			}
		}
	}
	return ret;
}
bool saveTargetAndCameras(const std::filesystem::path& path, const std::vector<SceneData>& scene)
{
	if (scene.size()<1)
	{
		LOG_ERR_OUT << "scene.size()<1";
		return false;
	}
	{
		std::fstream fout(path, std::ios::out);
		fout << "ply" << std::endl;
		fout << "format binary_little_endian 1.0" << std::endl;
		fout << "element vertex " << scene.size() << std::endl;//可以容纳32位数
		fout << "property float x" << std::endl;
		fout << "property float y" << std::endl;
		fout << "property float z" << std::endl;
		fout << "property uchar red" << std::endl;
		fout << "property uchar green" << std::endl;
		fout << "property uchar blue" << std::endl;
		fout << "property uint32 info" << std::endl;
		fout << "end_header" << std::endl;
		fout.close();
	}
	{
		std::fstream fout(path, std::ios::app | std::ios::binary);
		for (int i = 0; i < scene.size(); i++)
		{
			fout.write((const char*)&(scene[i].x), sizeof(float));
			fout.write((const char*)&(scene[i].y), sizeof(float));
			fout.write((const char*)&(scene[i].z), sizeof(float));
			fout.write((const char*)&(scene[i].r), sizeof(std::uint8_t));
			fout.write((const char*)&(scene[i].g), sizeof(std::uint8_t));
			fout.write((const char*)&(scene[i].b), sizeof(std::uint8_t));
			fout.write((const char*)&(scene[i].i), sizeof(std::uint32_t));
			 
		}
		fout.close();
	}
	return true;
}
template <typename T>
Json::Value vectorToJson(const std::vector<T>& vec) {
	Json::Value arr(Json::arrayValue);
	for (const auto& item : vec) {
		arr.append(item);
	}
	return arr;
}
bool writeResult(
	const std::filesystem::path& resultDir,
	const std::vector<Camera>& cameraList,
	const std::vector<Image>& imageList,
	const std::unordered_map<point3D_t, Eigen::Vector3d>& objPts,
	const std::unordered_map < image_t, struct Rigid3d>& poses)
{
	if (std::filesystem::exists(resultDir))
	{
		removeDirRecursive(resultDir);
	}
	std::filesystem::create_directories(resultDir);
	std::vector<SceneData> scene;
	UndistortCameraOptions undistortion_options;
	std::unordered_map<camera_t, Camera>undistortedCameraMap;
	std::unordered_map<std::string, BitmapColor<std::uint8_t>>labelColors;
	 
	std::fstream fPtsTxt(resultDir / "pts.txt", std::ios::out);
	Json::Value labelRoot;
	Json::Value viewRoot;
	Json::Value camerasRoot;
	{
		for (int i = 0; i < cameraList.size(); i++)
		{ 
			Json::Value camRoot;
			camRoot["cameraId"] = cameraList[i].camera_id;
			camRoot["model_id"] = static_cast<int>(cameraList[i].model_id);
			camRoot["height"] = cameraList[i].height;
			camRoot["width"] = cameraList[i].width;
			camRoot["params"] = vectorToJson(cameraList[i].params);
			camerasRoot.append(camRoot);
		}
	}
	std::filesystem::path ptsJsonPath = resultDir / "pts.json";
	Eigen::Vector3f objPtsRegionMax(-std::numeric_limits<float>::max(), -std::numeric_limits<float>::max(), -std::numeric_limits<float>::max());
	Eigen::Vector3f objPtsRegionMin(std::numeric_limits<float>::max(), std::numeric_limits<float>::max(), std::numeric_limits<float>::max());
	for (const auto& [ptId, pt] : objPts)
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
		scene.emplace_back(pt.x(), pt.y(), pt.z(), 255,255,255,255);
		if (objPtsRegionMin.x() > pt.x())objPtsRegionMin.x() = pt.x();
		if (objPtsRegionMin.y() > pt.y())objPtsRegionMin.y() = pt.y();
		if (objPtsRegionMin.z() > pt.z())objPtsRegionMin.z() = pt.z();
		if (objPtsRegionMax.x() < pt.x())objPtsRegionMax.x() = pt.x();
		if (objPtsRegionMax.y() < pt.y())objPtsRegionMax.y() = pt.y();
		if (objPtsRegionMax.z() < pt.z())objPtsRegionMax.z() = pt.z();
	}
	float cameraScale =  (objPtsRegionMax - objPtsRegionMin).maxCoeff()/20;
	std::vector<Eigen::Vector3d>cameraTs;
	cameraTs.reserve(poses.size());
	for (const auto&[imgId,Rt]: poses)
	{
		const Image& image1 = imageList[imgId];
		camera_t cameraId = image1.CameraId();
		std::filesystem::path originalPath(image1.Name());
		auto subDirName = originalPath.parent_path().filename().string();
		auto fileName = originalPath.filename().stem().string();
		auto newImgPath = resultDir / (subDirName + "@" + fileName + ".jpg");
		auto imgJsonPath = resultDir / (subDirName + "@" + fileName + ".json");
		if (undistortedCameraMap.count(cameraId)==0)
		{
			const Camera& camera1 = cameraList[cameraId];
			undistortedCameraMap[cameraId] = UndistortCamera(undistortion_options, camera1);
		}

		const Camera& distorted_camera = cameraList[cameraId];
		const Camera& undistorted_camera = undistortedCameraMap[cameraId];
		cameraTs.emplace_back(Rt.translation);
		Bitmap distorted_bitmap;
		distorted_bitmap.Read(image1.PicPath());
		Bitmap  undistorted_bitmap;//
		//WarpImageBetweenCameras(distorted_camera,
		//	undistorted_camera,
		//	distorted_bitmap,
		//	&undistorted_bitmap);

		Eigen::MatrixXi source_to_target_x_map;
		Eigen::MatrixXi source_to_target_y_map;
		WarpImageBetweenCameras(distorted_camera,
			undistorted_camera,
			distorted_bitmap,
			&undistorted_bitmap,
			source_to_target_x_map,
			source_to_target_y_map);

		distorted_bitmap.CloneMetadata(&undistorted_bitmap);
		undistorted_bitmap.Write(newImgPath.string());

		{
			for (const auto& d__ : image1.featPts)
			{
				int x = d__.second.x();
				int y = d__.second.y();
				int x2 = -1;
				int y2 = -1;
				if (x >= 0 && x < undistorted_camera.width && y >= 0 && y < undistorted_camera.height)
				{
					x2 = source_to_target_x_map(y, x);
					y2 = source_to_target_y_map(y, x);
				}
				const std::string&labelName = Image::keypointIndexToName[d__.first];
				if (labelColors.count(labelName)==0)
				{
					labelColors[labelName] = BitmapColor<std::uint8_t>::randColor();
				}
				undistorted_bitmap.circle(x2,y2,5, labelColors[labelName]);
			}
			undistorted_bitmap.Write(newImgPath.string()+".png");
		}

		newImgPath = std::filesystem::canonical(newImgPath);
		image1.writeRegisterJson(imgJsonPath, 1, newImgPath,
			undistorted_camera.params[0], undistorted_camera.params[1], undistorted_camera.params[2], undistorted_camera.params[3],
			undistorted_camera.height, undistorted_camera.width, Rt, source_to_target_x_map, source_to_target_y_map);

		std::vector<SceneData> cameraCloudPoint = getCameraCloudPoint(Rt, undistorted_camera.height, undistorted_camera.width, cameraScale, imgId);
		scene.reserve(scene.size() + cameraCloudPoint.size());
		scene.insert(scene.end(), cameraCloudPoint.begin(), cameraCloudPoint.end());


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

		Json::Value viewNode;
		viewNode["picPath"] = Json::Value(image1.PicPath());
		viewNode["jsonPath"] = Json::Value(image1.JsonPath());
		viewNode["name"] = Json::Value(fileName);
		viewNode["cameraId"] = cameraId;
		viewNode["imgId"] = imgId;
		viewNode["Qt"] = Json::Value();
		viewNode["Qt"].append(Rt.rotation.w());
		viewNode["Qt"].append(Rt.rotation.x());
		viewNode["Qt"].append(Rt.rotation.y());
		viewNode["Qt"].append(Rt.rotation.z());
		viewNode["Qt"].append(Rt.translation.x());
		viewNode["Qt"].append(Rt.translation.y());
		viewNode["Qt"].append(Rt.translation.z());
		viewRoot.append(viewNode);
	}
	std::vector<SceneData>cameraTrajectories = getCameraTrajectories(cameraTs);
	scene.reserve(scene.size() + cameraTrajectories.size());
	scene.insert(scene.end(), cameraTrajectories.begin(), cameraTrajectories.end());
	Json::StyledWriter sw;
	std::fstream fout(ptsJsonPath.string(), std::ios::out);
	{
		Json::Value rootNode;
		rootNode["obj"] = labelRoot;
		rootNode["cam"] = camerasRoot;
		rootNode["view"] = viewRoot;
		fout << sw.write(rootNode);
	}
	fout.close();
	fPtsTxt.close();
	saveTargetAndCameras(resultDir / "scene.ply", scene);
	return true;
}


