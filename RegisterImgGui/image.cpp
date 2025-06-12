#include <numeric>
#include <vector>
#include "image.h"
//#include "colmap/geometry/pose.h"
//#include "colmap/scene/projection.h"
 
Image::Image()
    : image_id_(kInvalidImageId),
    name_(""),
    camera_id_(kInvalidCameraId),
    camera_ptr_(nullptr),
    num_points3D_(0) {}



void Image::SetPoints2D(const std::map<point2D_t, Eigen::Vector2d>& featPts)
{
    if (featPts.size()==0)LOG_ERR_OUT << "error!!!";
    points2D_.resize(featPts.size());
    std::vector<point2D_t>featIds;
    featIds.reserve(featPts.size());
    for (const auto&d: featPts)
    {
        featIds.emplace_back(d.first);
    }
    std::sort(featIds.begin(), featIds.end());
    for (int point2D_idx = 0; point2D_idx < featIds.size(); ++point2D_idx) {
        points2D_[point2D_idx] = featPts.at(featIds[point2D_idx]);
    }
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

std::map<std::string, int>Image::keypointNameToIndx;
std::map<int, std::string>Image::keypointIndexToName;
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
