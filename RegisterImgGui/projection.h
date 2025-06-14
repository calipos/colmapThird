#ifndef _PROJECTION_H_
#define _PROJECTION_H_


#include "rigid3.h"
#include "camera.h"
#include "eigen_alignment.h"

#include <limits>
#include <vector>

#include <Eigen/Core>
#include <Eigen/Geometry>


// Calculate the reprojection error.
//
// The reprojection error is the Euclidean distance between the observation
// in the image and the projection of the 3D point into the image. If the
// 3D point is behind the camera, then this function returns DBL_MAX.
double CalculateSquaredReprojectionError(const Eigen::Vector2d& point2D,
    const Eigen::Vector3d& point3D,
    const Rigid3d& cam_from_world,
    const Camera& camera);
double CalculateSquaredReprojectionError(
    const Eigen::Vector2d& point2D,
    const Eigen::Vector3d& point3D,
    const Eigen::Matrix3x4d& cam_from_world,
    const Camera& camera);

// Calculate the angular error.
//
// The angular error is the angle between the observed viewing ray and the
// actual viewing ray from the camera center to the 3D point.
double CalculateAngularError(const Eigen::Vector2d& point2D,
    const Eigen::Vector3d& point3D,
    const Rigid3d& cam_from_world,
    const Camera& camera);
double CalculateAngularError(const Eigen::Vector2d& point2D,
    const Eigen::Vector3d& point3D,
    const Eigen::Matrix3x4d& cam_from_world,
    const Camera& camera);

// Calculate angulate error using normalized image points.
//
// The angular error is the angle between the observed viewing ray and the
// actual viewing ray from the camera center to the 3D point.
double CalculateNormalizedAngularError(const Eigen::Vector2d& point2D,
    const Eigen::Vector3d& point3D,
    const Rigid3d& cam_from_world);
double CalculateNormalizedAngularError(const Eigen::Vector2d& point2D,
    const Eigen::Vector3d& point3D,
    const Eigen::Matrix3x4d& cam_from_world);

// Check if 3D point passes cheirality constraint,
// i.e. it lies in front of the camera and not in the image plane.
//
// @param cam_from_world  3x4 projection matrix.
// @param point3D         3D point as 3x1 vector.
//
// @return                True if point lies in front of camera.
bool HasPointPositiveDepth(const Eigen::Matrix3x4d& cam_from_world,
    const Eigen::Vector3d& point3D);

#endif // !_PROJECTION_H_
