#include "reconstruction.h"
#include "image.h"
#include "camera.h"



Reconstruction::Reconstruction() : max_point3D_id_(0) {}

Reconstruction::Reconstruction(const Reconstruction& other)
    : cameras_(other.cameras_),
    images_(other.images_),
    points3D_(other.points3D_),
    reg_image_ids_(other.reg_image_ids_),
    max_point3D_id_(other.max_point3D_id_) {
    for (auto& [_, image] : images_) {
        image.ResetCameraPtr();
        image.SetCameraPtr(&Camera(image.CameraId()));
    }
}

Reconstruction& Reconstruction::operator=(const Reconstruction& other) {
    if (this != &other) {
        cameras_ = other.cameras_;
        images_ = other.images_;
        points3D_ = other.points3D_;
        reg_image_ids_ = other.reg_image_ids_;
        max_point3D_id_ = other.max_point3D_id_;
        for (auto& [_, image] : images_) {
            image.ResetCameraPtr();
            image.SetCameraPtr(&Camera(image.CameraId()));
        }
    }
    return *this;
}

std::unordered_set<point3D_t> Reconstruction::Point3DIds() const {
    std::unordered_set<point3D_t> point3D_ids;
    point3D_ids.reserve(points3D_.size());

    for (const auto& point3D : points3D_) {
        point3D_ids.insert(point3D.first);
    }

    return point3D_ids;
}


void Reconstruction::TearDown() {
    // Remove all not yet registered images.
    std::unordered_set<camera_t> keep_camera_ids;
    for (auto it = images_.begin(); it != images_.end();) {
        if (IsImageRegistered(it->first)) {
            keep_camera_ids.insert(it->second.CameraId());
            ++it;
        }
        else {
            it = images_.erase(it);
        }
    }

    // Remove all unused cameras.
    for (auto it = cameras_.begin(); it != cameras_.end();) {
        if (keep_camera_ids.count(it->first) == 0) {
            it = cameras_.erase(it);
        }
        else {
            ++it;
        }
    }

    // Compress tracks.
    for (auto& point3D : points3D_) {
        //point3D.second.track.Compress();
    }
}

void Reconstruction::AddCamera(const struct Camera& camera) {
    const camera_t camera_id = camera.camera_id;
    (camera.VerifyParams());
    (cameras_.emplace(camera_id, std::move(camera)).second);
}

void Reconstruction::AddImage(class Image &image) {
    auto& camera = Camera(image.CameraId());
    if (image.HasCameraPtr()) {
    }
    else {
        image.SetCameraPtr(&camera);
    }
    const image_t image_id = image.ImageId();
    const bool is_registered = image.HasPose();
    images_.emplace(image_id, std::move(image)).second;
    if (is_registered) {
        //RegisterImage(image_id);
    }
}

void Reconstruction::AddPoint3D(const point3D_t&point3D_id,
    const Eigen::Vector3d& point3D) {
    max_point3D_id_ = std::max(max_point3D_id_, point3D_id);
    //points3D_.emplace(point3D_id, std::move(point3D)).second;
}
 