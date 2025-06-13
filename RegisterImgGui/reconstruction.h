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
#endif // !_RECONSTRUCTION_H_

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
    inline const struct Camera& Camera(camera_t camera_id) const;
    inline const class Image& Image(image_t image_id) const;
    inline const Eigen::Vector3d& Point3D(point3D_t point3D_id) const;

    // Get mutable objects.
    inline struct Camera& Camera(camera_t camera_id);
    inline class Image& Image(image_t image_id);
    inline Eigen::Vector3d& Point3D(point3D_t point3D_id);

    // Get reference to all objects.
    inline const std::unordered_map<camera_t, struct Camera>& Cameras() const;
    inline const std::unordered_map<image_t, class Image>& Images() const;
    inline const std::set<image_t>& RegImageIds() const;
    inline const std::unordered_map<point3D_t, Eigen::Vector3d>& Points3D() const;

    // Identifiers of all 3D points.
    std::unordered_set<point3D_t> Point3DIds() const;

    // Check whether specific object exists.
    inline bool ExistsCamera(camera_t camera_id) const;
    inline bool ExistsImage(image_t image_id) const;
    inline bool ExistsPoint3D(point3D_t point3D_id) const;


    // Finalize the Reconstruction after the reconstruction has finished.
    // Once a scene has been finalized, it cannot be used for reconstruction.
    // This removes all not yet registered images and unused cameras, in order to
    // save memory.
    void TearDown();

    // Add new camera. There is only one camera per image, while multiple images
    // might be taken by the same camera.
    void AddCamera(struct Camera camera);

    // Add new image. Its camera must have been added before. If its camera object
    // is unset, it will be automatically populated from the added cameras.
    void AddImage(class Image image);

    // Add new 3D point with known ID.
    void AddPoint3D(point3D_t point3D_id, Eigen::Vector3d point3D);



    // Merge two 3D points and return new identifier of new 3D point.
    // The location of the merged 3D point is a weighted average of the two
    // original 3D point's locations according to their track lengths.
    point3D_t MergePoints3D(point3D_t point3D_id1, point3D_t point3D_id2);

    // Delete a 3D point, and all its references in the observed images.
    void DeletePoint3D(point3D_t point3D_id);

    // Delete one observation from an image and the corresponding 3D point.
    // Note that this deletes the entire 3D point, if the track has two elements
    // prior to calling this method.
    void DeleteObservation(image_t image_id, point2D_t point2D_idx);

    // Delete all 2D points of all images and all 3D points.
    void DeleteAllPoints2DAndPoints3D();

    // Register an existing image.
    void RegisterImage(image_t image_id);

    // De-register an existing image, and all its references.
    void DeRegisterImage(image_t image_id);

    // Check if image is registered.
    inline bool IsImageRegistered(image_t image_id) const;

    // Normalize scene by scaling and translation to avoid degenerate
    // visualization after bundle adjustment and to improve numerical
    // stability of algorithms.
    //
    // Translates scene such that the mean of the camera centers or point
    // locations are at the origin of the coordinate system.
    //
    // Scales scene such that the minimum and maximum camera centers are at the
    // given `extent`, whereas `p0` and `p1` determine the minimum and
    // maximum percentiles of the camera centers considered.
    Sim3d Normalize(bool fixed_scale = false,
        double extent = 10.0,
        double p0 = 0.1,
        double p1 = 0.9,
        bool use_images = true);

    // Compute the centroid of the 3D points
    Eigen::Vector3d ComputeCentroid(double p0 = 0.1, double p1 = 0.9) const;

    // Compute the bounding box corners of the 3D points
    std::pair<Eigen::Vector3d, Eigen::Vector3d> ComputeBoundingBox(
        double p0 = 0.0, double p1 = 1.0) const;

    // Apply the 3D similarity transformation to all images and points.
    void Transform(const Sim3d& new_from_old_world);

    // Creates a cropped reconstruction using the input bounds as corner points
    // of the bounding box containing the included 3D points of the new
    // reconstruction. Only the cameras and images of the included points are
    // registered.
    Reconstruction Crop(
        const std::pair<Eigen::Vector3d, Eigen::Vector3d>& bbox) const;

    // Find specific image by name. Note that this uses linear search.
    const class Image* FindImageWithName(const std::string& name) const;

    // Find images that are both present in this and the given reconstruction.
    // Matching of images is performed based on common image names.
    std::vector<std::pair<image_t, image_t>> FindCommonRegImageIds(
        const Reconstruction& other) const;

    // Compute statistics for scene.
    size_t ComputeNumObservations() const;
    double ComputeMeanTrackLength() const;
    double ComputeMeanObservationsPerRegImage() const;
    double ComputeMeanReprojectionError() const;

    // Updates mean reprojection errors for all 3D points.
    void UpdatePoint3DErrors();

    // Read data from text or binary file. Prefer binary data if it exists.
    void Read(const std::string& path);
    void Write(const std::string& path) const;

    // Read data from binary/text file.
    void ReadText(const std::string& path);
    void ReadBinary(const std::string& path);

    // Write data from binary/text file.
    void WriteText(const std::string& path) const;
    void WriteBinary(const std::string& path) const;

    // Convert 3D points in reconstruction to PLY point cloud.
    std::vector<PlyPoint> ConvertToPLY() const;

    // Import from other data formats. Note that these import functions are
    // only intended for visualization of data and unusable for reconstruction.
    void ImportPLY(const std::string& path);
    void ImportPLY(const std::vector<PlyPoint>& ply_points);

    // Extract colors for 3D points of given image. Colors will be extracted
    // only for 3D points which are completely black.
    //
    // @param image_id      Identifier of the image for which to extract colors.
    // @param path          Absolute or relative path to root folder of image.
    //                      The image path is determined by concatenating the
    //                      root path and the name of the image.
    //
    // @return              True if image could be read at given path.
    bool ExtractColorsForImage(image_t image_id, const std::string& path);

    // Extract colors for all 3D points by computing the mean color of all images.
    //
    // @param path          Absolute or relative path to root folder of image.
    //                      The image path is determined by concatenating the
    //                      root path and the name of the image.
    void ExtractColorsForAllImages(const std::string& path);

    // Create all image sub-directories in the given path.
    void CreateImageDirs(const std::string& path) const;

private:
    std::tuple<Eigen::Vector3d, Eigen::Vector3d, Eigen::Vector3d>
        ComputeBoundsAndCentroid(double p0, double p1, bool use_images) const;

    std::unordered_map<camera_t, struct Camera> cameras_;
    std::unordered_map<image_t, class Image> images_;
    std::unordered_map<point3D_t, Eigen::Vector3d> points3D_;

    // { image_id, ... } where `images_.at(image_id).registered == true`.
    std::set<image_t> reg_image_ids_;

    // Total number of added 3D points, used to generate unique identifiers.
    point3D_t max_point3D_id_;
};

std::ostream& operator<<(std::ostream& stream,
    const Reconstruction& reconstruction);

////////////////////////////////////////////////////////////////////////////////
// Implementation
////////////////////////////////////////////////////////////////////////////////

size_t Reconstruction::NumCameras() const { return cameras_.size(); }

size_t Reconstruction::NumImages() const { return images_.size(); }

size_t Reconstruction::NumRegImages() const { return reg_image_ids_.size(); }

size_t Reconstruction::NumPoints3D() const { return points3D_.size(); }

const struct Camera& Reconstruction::Camera(const camera_t camera_id) const {
    try {
        return cameras_.at(camera_id);
    }
    catch (const std::out_of_range&) {
        LOG_ERR_OUT << "Camera with ID " << camera_id << " does not exist";
        throw std::out_of_range("");
    }
}

const class Image& Reconstruction::Image(const image_t image_id) const {
    try {
        return images_.at(image_id);
    }
    catch (const std::out_of_range&) {
        LOG_ERR_OUT << "Image with ID " << image_id << " does not exist";
        throw std::out_of_range("");
    }
}

const Eigen::Vector3d& Reconstruction::Point3D(
    const point3D_t point3D_id) const {
    try {
        return points3D_.at(point3D_id);
    }
    catch (const std::out_of_range&) {
        LOG_ERR_OUT << "Point3D with ID " << point3D_id << " does not exist";
        throw std::out_of_range("");
    }
}

struct Camera& Reconstruction::Camera(const camera_t camera_id) {
    try {
        return cameras_.at(camera_id);
    }
    catch (const std::out_of_range&) {
        LOG_ERR_OUT << "Camera with ID " << camera_id << " does not exist";
        throw std::out_of_range("");
    }
}

class Image& Reconstruction::Image(const image_t image_id) {
    try {
        return images_.at(image_id);
    }
    catch (const std::out_of_range&) {
        LOG_ERR_OUT << "Image with ID " << image_id << " does not exist";
        throw std::out_of_range("");
    }
}

Eigen::Vector3d& Reconstruction::Point3D(const point3D_t point3D_id) {
    try {
        return points3D_.at(point3D_id);
    }
    catch (const std::out_of_range&) {
        LOG_ERR_OUT << "Point3D with ID " << point3D_id << " does not exist";
        throw std::out_of_range("");
    }
}

const std::unordered_map<camera_t, Camera>& Reconstruction::Cameras() const {
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

bool Reconstruction::ExistsCamera(const camera_t camera_id) const {
    return cameras_.find(camera_id) != cameras_.end();
}

bool Reconstruction::ExistsImage(const image_t image_id) const {
    return images_.find(image_id) != images_.end();
}

bool Reconstruction::ExistsPoint3D(const point3D_t point3D_id) const {
    return points3D_.find(point3D_id) != points3D_.end();
}

bool Reconstruction::IsImageRegistered(const image_t image_id) const {
    return reg_image_ids_.find(image_id) != reg_image_ids_.end();
}
