#ifndef POSE_H_
#define POSE_H_
#include <vector>
#include <Eigen/Eigenvalues>
#include "eigen_alignment.h"
#include "rigid3.h"
#include "sim3.h"
// Compute the closes rotation matrix with the closest Frobenius norm by setting
// the singular values of the given matrix to 1.
Eigen::Matrix3d ComputeClosestRotationMatrix(const Eigen::Matrix3d& matrix);

// Decompose projection matrix into intrinsic camera matrix, rotation matrix and
// translation vector. Returns false if decomposition fails.
bool DecomposeProjectionMatrix(const Eigen::Matrix3x4d& proj_matrix,
    Eigen::Matrix3d* K,
    Eigen::Matrix3d* R,
    Eigen::Vector3d* T);

// Compose the skew symmetric cross product matrix from a vector.
Eigen::Matrix3d CrossProductMatrix(const Eigen::Vector3d& vector);

// Convert 3D rotation matrix to Euler angles.
//
// The convention `R = Rx * Ry * Rz` is used,
// using a right-handed coordinate system.
//
// @param R              3x3 rotation matrix.
// @param rx, ry, rz     Euler angles in radians.
void RotationMatrixToEulerAngles(const Eigen::Matrix3d& R,
    double* rx,
    double* ry,
    double* rz);

// Convert Euler angles to 3D rotation matrix.
//
// The convention `R = Rz * Ry * Rx` is used,
// using a right-handed coordinate system.
//
// @param rx, ry, rz     Euler angles in radians.
//
// @return               3x3 rotation matrix.
Eigen::Matrix3d EulerAnglesToRotationMatrix(double rx, double ry, double rz);

// Compute the weighted average of multiple Quaternions according to:
//
//    Markley, F. Landis, et al. "Averaging quaternions."
//    Journal of Guidance, Control, and Dynamics 30.4 (2007): 1193-1197.
//
// @param quats         The Quaternions to be averaged.
// @param weights       Non-negative weights.
//
// @return              The average Quaternion.
Eigen::Quaterniond AverageQuaternions(
    const std::vector<Eigen::Quaterniond>& quats,
    const std::vector<double>& weights);

// Linearly interpolate camera pose.
struct Rigid3d InterpolateCameraPoses(const struct Rigid3d& cam_from_world1,
    const struct Rigid3d& cam_from_world2,
    double t);

// Perform cheirality constraint test, i.e., determine which of the triangulated
// correspondences lie in front of both cameras.
//
// @param cam2_from_cam1  Relative camera transformation.
// @param points1         First set of corresponding points.
// @param points2         Second set of corresponding points.
// @param points3D        Points that lie in front of both cameras.
bool CheckCheirality(const struct Rigid3d& cam2_from_cam1,
    const std::vector<Eigen::Vector2d>& points1,
    const std::vector<Eigen::Vector2d>& points2,
    std::vector<Eigen::Vector3d>* points3D);

struct Rigid3d TransformCameraWorld(const struct Sim3d& new_from_old_world,
    const struct Rigid3d& cam_from_world);


#endif // !POSE_H_
