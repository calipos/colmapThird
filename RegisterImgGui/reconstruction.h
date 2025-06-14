#ifndef _RECONSTRUCTION_H_
#define _RECONSTRUCTION_H_
#include "sim3.h"
#include "camera.h"
#include "image.h"
#include "eigen_alignment.h"
#include "types.h"
#include <memory>
#include <set>
#include <tuple>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <Eigen/Core>

struct PlyPoint;
// Reconstruction class holds all information about a single reconstructed
// model. It is used by the mapping and bundle adjustment classes and can be
// written to and read from disk.
class Reconstruction {
public:
    Reconstruction();

    // Copy construct/assign. Updates camera pointers.
    Reconstruction(const Reconstruction& other);
    Reconstruction& operator=(const Reconstruction& other);

    // Get number of objects.
    inline size_t NumCameras() const;
    inline size_t NumImages() const;
    inline size_t NumRegImages() const;
    inline size_t NumPoints3D() const;

    // Get const objects.
    inline const struct Camera& Camera(const camera_t&camera_id) const;
    inline const class Image& Image(const image_t& image_id) const;
    inline const Eigen::Vector3d& Point3D(const point3D_t&point3D_id) const;

    // Get mutable objects.
    inline struct Camera& Camera(const camera_t& camera_id);
    inline class Image& Image(const image_t&image_id);
    inline Eigen::Vector3d& Point3D(const point3D_t& point3D_id);

    // Get reference to all objects.
    inline const std::unordered_map<camera_t, struct Camera>& Cameras() const;
    inline const std::unordered_map<image_t, class Image>& Images() const;
    inline const std::set<image_t>& RegImageIds() const;
    inline const std::unordered_map<point3D_t, Eigen::Vector3d>& Points3D() const;

    // Identifiers of all 3D points.
    std::unordered_set<point3D_t> Point3DIds() const;

    // Check whether specific object exists.
    inline bool ExistsCamera(const camera_t&camera_id) const;
    inline bool ExistsImage(const image_t& image_id) const;
    inline bool ExistsPoint3D(const point3D_t&point3D_id) const;


    // Finalize the Reconstruction after the reconstruction has finished.
    // Once a scene has been finalized, it cannot be used for reconstruction.
    // This removes all not yet registered images and unused cameras, in order to
    // save memory.
    void TearDown();

    // Add new camera. There is only one camera per image, while multiple images
    // might be taken by the same camera.
    void AddCamera(const struct Camera&camera);

    // Add new image. Its camera must have been added before. If its camera object
    // is unset, it will be automatically populated from the added cameras.
    void AddImage(class Image& image);

    // Add new 3D point with known ID.
    void AddPoint3D(const point3D_t&point3D_id, const Eigen::Vector3d& point3D);




    // Check if image is registered.
    inline bool IsImageRegistered(const image_t& image_id) const;



private:

    std::unordered_map<camera_t, struct Camera> cameras_;
    std::unordered_map<image_t, class Image> images_;
    std::unordered_map<point3D_t, Eigen::Vector3d> points3D_;

    // { image_id, ... } where `images_.at(image_id).registered == true`.
    std::set<image_t> reg_image_ids_;

    // Total number of added 3D points, used to generate unique identifiers.
    point3D_t max_point3D_id_;
};


////////////////////////////////////////////////////////////////////////////////
// Implementation
////////////////////////////////////////////////////////////////////////////////

size_t Reconstruction::NumCameras() const { return cameras_.size(); }

size_t Reconstruction::NumImages() const { return images_.size(); }

size_t Reconstruction::NumRegImages() const { return reg_image_ids_.size(); }

size_t Reconstruction::NumPoints3D() const { return points3D_.size(); }

const struct Camera& Reconstruction::Camera(const camera_t&camera_id) const {
    try {
        return cameras_.at(camera_id);
    }
    catch (const std::out_of_range&) {
        LOG_ERR_OUT << "Camera with ID " << camera_id << " does not exist";
        throw std::out_of_range("");
    }
}

const Image& Reconstruction::Image(const image_t &image_id) const {
    try {
        return images_.at(image_id);
    }
    catch (const std::out_of_range&) {
        LOG_ERR_OUT << "Image with ID " << image_id << " does not exist";
        throw std::out_of_range("");
    }
}

const Eigen::Vector3d& Reconstruction::Point3D(
    const point3D_t&point3D_id) const {
    try {
        return points3D_.at(point3D_id);
    }
    catch (const std::out_of_range&) {
        LOG_ERR_OUT << "Point3D with ID " << point3D_id << " does not exist";
        throw std::out_of_range("");
    }
}

struct Camera& Reconstruction::Camera(const camera_t&camera_id) {
    try {
        return cameras_.at(camera_id);
    }
    catch (const std::out_of_range&) {
        LOG_ERR_OUT << "Camera with ID " << camera_id << " does not exist";
        throw std::out_of_range("");
    }
}

class Image& Reconstruction::Image(const image_t& image_id) {
    try {
        return images_.at(image_id);
    }
    catch (const std::out_of_range&) {
        LOG_ERR_OUT << "Image with ID " << image_id << " does not exist";
        throw std::out_of_range("");
    }
}

Eigen::Vector3d& Reconstruction::Point3D(const point3D_t&point3D_id) {
    try {
        return points3D_.at(point3D_id);
    }
    catch (const std::out_of_range&) {
        LOG_ERR_OUT << "Point3D with ID " << point3D_id << " does not exist";
        throw std::out_of_range("");
    }
}

const std::unordered_map<camera_t, struct Camera>& Reconstruction::Cameras() const {
    return cameras_;
}

const std::unordered_map<image_t, class Image>& Reconstruction::Images() const {
    return images_;
}

const std::set<image_t>& Reconstruction::RegImageIds() const {
    return reg_image_ids_;
}

const std::unordered_map<point3D_t, Eigen::Vector3d>& Reconstruction::Points3D() const {
    return points3D_;
}

bool Reconstruction::ExistsCamera(const camera_t& camera_id) const {
    return cameras_.find(camera_id) != cameras_.end();
}

bool Reconstruction::ExistsImage(const image_t& image_id) const {
    return images_.find(image_id) != images_.end();
}

bool Reconstruction::ExistsPoint3D(const point3D_t& point3D_id) const {
    return points3D_.find(point3D_id) != points3D_.end();
}

bool Reconstruction::IsImageRegistered(const image_t& image_id) const {
    return reg_image_ids_.find(image_id) != reg_image_ids_.end();
}

#endif // !_RECONSTRUCTION_H_