#ifndef _TRIANGULATION_H_
#define _TRIANGULATION_H_
#include <vector>
#include <Eigen/Core>
#include "types.h"
#include "eigen_alignment.h"
// Triangulate 3D point from corresponding image point observations.
//
// Implementation of the direct linear transform triangulation method in
//   R. Hartley and A. Zisserman, Multiple View Geometry in Computer Vision,
//   Cambridge Univ. Press, 2003.
//
// @param cam_from_world1   Projection matrix of the first image as 3x4 matrix.
// @param cam_from_world2   Projection matrix of the second image as 3x4 matrix.
// @param point1            Corresponding 2D point in first image.
// @param point2            Corresponding 2D point in second image.
// @param point3D           Triangulated 3D point.
//
// @return                  Whether triangulation was successful.
bool TriangulatePoint(const Eigen::Matrix3x4d& cam_from_world1,
    const Eigen::Matrix3x4d& cam_from_world2,
    const Eigen::Vector2d& point1,
    const Eigen::Vector2d& point2,
    Eigen::Vector3d* point3D);

// Triangulate point from multiple views minimizing the L2 error.
//
// @param cams_from_world   Projection matrices of multi-view observations.
// @param points            Image observations of multi-view observations.
// @param point3D           Triangulated 3D point.
//
// @return                  Whether triangulation was successful.
bool TriangulateMultiViewPoint(
    const std::vector<Eigen::Matrix3x4d>& cams_from_world,
    const std::vector<Eigen::Vector2d>& points,
    Eigen::Vector3d* point3D);

// Triangulate optimal 3D point from corresponding image point observations by
// finding the optimal image observations.
//
// Note that camera poses should be very good in order for this method to yield
// good results. Otherwise just use `TriangulatePoint`.
//
// Implementation of the method described in
//   P. Lindstrom, "Triangulation Made Easy," IEEE Computer Vision and Pattern
//   Recognition 2010, pp. 1554-1561, June 2010.
//
// @param cam_from_world1   Projection matrix of the first image as 3x4 matrix.
// @param cam_from_world2   Projection matrix of the second image as 3x4 matrix.
// @param point1            Corresponding 2D point in first image.
// @param point2            Corresponding 2D point in second image.
// @param point3D           Triangulated 3D point.
//
// @return                  Whether triangulation was successful.
bool TriangulateOptimalPoint(const Eigen::Matrix3x4d& cam_from_world1,
    const Eigen::Matrix3x4d& cam_from_world2,
    const Eigen::Vector2d& point1,
    const Eigen::Vector2d& point2,
    Eigen::Vector3d* point3D);

// Calculate angle in radians between the two rays of a triangulated point.
double CalculateTriangulationAngle(const Eigen::Vector3d& proj_center1,
    const Eigen::Vector3d& proj_center2,
    const Eigen::Vector3d& point3D);
std::vector<double> CalculateTriangulationAngles(
    const Eigen::Vector3d& proj_center1,
    const Eigen::Vector3d& proj_center2,
    const std::vector<Eigen::Vector3d>& points3D);

#endif // !_TRIANGULATION_H_
