#include <numeric>
#include <vector>
#include <fstream>
#include "image.h"
#include "json/json.h"
//#include "colmap/geometry/pose.h"
//#include "colmap/scene/projection.h"
 
Image::Image()
    : image_id_(kInvalidImageId),
    name_(""),
    camera_id_(kInvalidCameraId),
    camera_ptr_(nullptr),
    num_points3D_(0) {}



bool Image::SetPoints2D(const std::map<point2D_t, Eigen::Vector2d>& featPts)
{
    if (featPts.size()==0)LOG_ERR_OUT << "error!!!";
    points2D_.resize(featPts.size(), Eigen::Vector2d(std::numeric_limits<double>::quiet_NaN(), std::numeric_limits<double>::quiet_NaN()));


    {
        std::vector<point2D_t>idSet;
        idSet.reserve(featPts.size());
        for (const auto& d : featPts)
        {
            idSet.emplace_back(d.first);
        }
        int maxId = *(std::max_element(idSet.begin(), idSet.end()));
        maxId = abs(maxId);
        points2D_.resize(maxId + 1, Eigen::Vector2d(std::numeric_limits<double>::quiet_NaN(), std::numeric_limits<double>::quiet_NaN()));
    }
    for (const auto& d : featPts)
    {
        if (d.first<0 || d.first>points2D_.size())
        {
            return false;
        }
        points2D_[d.first] = d.second;
    }
    return true;
    //std::vector<point2D_t>featIds;
    //featIds.reserve(featPts.size());
    //for (const auto&d: featPts)
    //{
    //    featIds.emplace_back(d.first);
    //}
    //std::sort(featIds.begin(), featIds.end());
    //for (int point2D_idx = 0; point2D_idx < featIds.size(); ++point2D_idx) {
    //    points2D_[featIds[point2D_idx]] = featPts.at(featIds[point2D_idx]);
    //}

}
int Image::writeRegisterJson(const std::filesystem::path&imgJsonPath,
    const int& version, const std::filesystem::path& imgPath,
    const double& fx, const double& fy,
    const double& cx, const double& cy,
    const int& imgHeight, const int& imgWidth,
    const struct Rigid3d& Rt,
    const Eigen::MatrixXi& source_to_target_x_map,
    const Eigen::MatrixXi& source_to_target_y_map)const
{
    Json::Value labelRoot;
    labelRoot["version"] = Json::Value(std::to_string(version));
    labelRoot["imagePath"] = Json::Value(imgPath.string());
    labelRoot["fx"] = Json::Value(fx);
    labelRoot["fy"] = Json::Value(fy);
    labelRoot["cx"] = Json::Value(cx);
    labelRoot["cy"] = Json::Value(cy);
    labelRoot["width"] = Json::Value(imgWidth);
    labelRoot["height"] = Json::Value(imgHeight);
    for (const auto& d__ : this->featPts)
    {
        int x = d__.second.x();
        int y = d__.second.y();
        int x2 = -1;
        int y2 = -1;
        if (x >= 0 && x < imgWidth && y >= 0 && y < imgHeight)
        {
            x2 = source_to_target_x_map(y, x);
            y2 = source_to_target_y_map(y, x);
        }
        Json::Value undistortedFeatPt;
        undistortedFeatPt.append(x2);
        undistortedFeatPt.append(y2);
        std::string ptName = "undistortedFeat_" + Image::keypointIndexToName[d__.first];
        labelRoot[ptName] = undistortedFeatPt;
    }
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

    return 0;
}
void Image::SetPoint3DForPoint2D(const point2D_t point2D_idx,
    const point3D_t point3D_id) {
    Eigen::Vector2d& point2D = points2D_.at(point2D_idx);
}

void Image::ResetPoint3DForPoint2D(const point2D_t point2D_idx) {
    Eigen::Vector2d& point2D = points2D_.at(point2D_idx);
}


Eigen::Vector3d Image::ProjectionCenter() const {
    return CamFromWorld().rotation.inverse() * -CamFromWorld().translation;
}

Eigen::Vector3d Image::ViewingDirection() const {
    return CamFromWorld().rotation.toRotationMatrix().row(2);
}

std::pair<bool, Eigen::Vector2d> Image::ProjectPoint(
    const Eigen::Vector3d& point3D) const {
    const Eigen::Vector3d point3D_in_cam = CamFromWorld() * point3D;
    if (point3D_in_cam.z() < std::numeric_limits<double>::epsilon()) {
        return { false, Eigen::Vector2d() };
    }
    return { true, camera_ptr_->ImgFromCam(point3D_in_cam.hnormalized()) };
}

std::map<std::string, point2D_t>Image::keypointNameToIndx;
std::map<point2D_t, std::string>Image::keypointIndexToName;
std::ostream& operator<<(std::ostream& stream, const Image& image) {
    stream << "Image(image_id="
        << (image.ImageId() != kInvalidImageId
            ? std::to_string(image.ImageId())
            : "Invalid");
    if (!image.HasCameraPtr()) {
        stream << ", camera_id="
            << (image.HasCameraId() ? std::to_string(image.CameraId())
                : "Invalid");
    }
    else {
        stream << ", camera=Camera(camera_id=" << std::to_string(image.CameraId())
            << ")";
    }
    stream << ", name=\"" << image.Name() << "\""
        << ", has_pose=" << image.HasPose()
        << ", triangulated=" << image.NumPoints3D() << "/"
        << image.NumPoints2D() << ")";
    return stream;
}
