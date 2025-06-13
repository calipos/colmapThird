#include <iostream>
#include <Eigen/Dense>
#include <algorithm>
#include "colmath.h"
#include "triangulation.h"
#include "log.h"
#include "rigid3.h"
#include "essential_matrix.h"

bool TriangulatePoint(const Eigen::Matrix3x4d& cam1_from_world,
    const Eigen::Matrix3x4d& cam2_from_world,
    const Eigen::Vector2d& point1,
    const Eigen::Vector2d& point2,
    Eigen::Vector3d* xyz) {
    if (nullptr== xyz)
    {
        LOG_ERR_OUT << "nullptr== xyz";
        return false;
    }
    Eigen::Matrix4d A;
    A.row(0) = point1(0) * cam1_from_world.row(2) - cam1_from_world.row(0);
    A.row(1) = point1(1) * cam1_from_world.row(2) - cam1_from_world.row(1);
    A.row(2) = point2(0) * cam2_from_world.row(2) - cam2_from_world.row(0);
    A.row(3) = point2(1) * cam2_from_world.row(2) - cam2_from_world.row(1);
    const Eigen::JacobiSVD<Eigen::Matrix4d> svd(A, Eigen::ComputeFullV);
#if EIGEN_VERSION_AT_LEAST(3, 4, 0)
    if (svd.info() != Eigen::Success) {
        return false;
    }
#endif
    if (svd.matrixV()(3, 3) == 0) {
        return false;
    }
    *xyz = svd.matrixV().col(3).hnormalized();
    return true;
}

bool TriangulateMultiViewPoint(
    const std::vector<Eigen::Matrix3x4d>& cams_from_world,
    const std::vector<Eigen::Vector2d>& points,
    Eigen::Vector3d* xyz) 
{
    if (cams_from_world.size() != points.size())
    {
        LOG_ERR_OUT << "cams_from_world.size()!= points.size()";
        return false;
    }
    if (nullptr == xyz)
    {
        LOG_ERR_OUT << "nullptr== xyz";
        return false;
    }
    Eigen::Matrix4d A = Eigen::Matrix4d::Zero();
    for (size_t i = 0; i < points.size(); i++) {
        const Eigen::Vector3d point = points[i].homogeneous().normalized();
        const Eigen::Matrix3x4d term =
            cams_from_world[i] - point * point.transpose() * cams_from_world[i];
        A += term.transpose() * term;
    }
    const Eigen::SelfAdjointEigenSolver<Eigen::Matrix4d> eigen_solver(A);
    if (eigen_solver.info() != Eigen::Success ||
        eigen_solver.eigenvectors()(3, 0) == 0) {
        return false;
    }
    *xyz = eigen_solver.eigenvectors().col(0).hnormalized();
    return true;
}

bool TriangulateOptimalPoint(const Eigen::Matrix3x4d& cam1_from_world_mat,
    const Eigen::Matrix3x4d& cam2_from_world_mat,
    const Eigen::Vector2d& point1,
    const Eigen::Vector2d& point2,
    Eigen::Vector3d* xyz) {
    const Rigid3d cam1_from_world(
        Eigen::Quaterniond(cam1_from_world_mat.leftCols<3>()),
        cam1_from_world_mat.col(3));
    const Rigid3d cam2_from_world(
        Eigen::Quaterniond(cam2_from_world_mat.leftCols<3>()),
        cam2_from_world_mat.col(3));
    const Rigid3d cam2_from_cam1 = cam2_from_world * Inverse(cam1_from_world);
    const Eigen::Matrix3d E = EssentialMatrixFromPose(cam2_from_cam1);

    Eigen::Vector2d optimal_point1;
    Eigen::Vector2d optimal_point2;
    FindOptimalImageObservations(
        E, point1, point2, &optimal_point1, &optimal_point2);

    return TriangulatePoint(cam1_from_world_mat,
        cam2_from_world_mat,
        optimal_point1,
        optimal_point2,
        xyz);
}

double CalculateTriangulationAngle(const Eigen::Vector3d& proj_center1,
    const Eigen::Vector3d& proj_center2,
    const Eigen::Vector3d& point3D) {
    const double baseline_length_squared =
        (proj_center1 - proj_center2).squaredNorm();

    const double ray_length_squared1 = (point3D - proj_center1).squaredNorm();
    const double ray_length_squared2 = (point3D - proj_center2).squaredNorm();

    // Using "law of cosines" to compute the enclosing angle between rays.
    const double denominator =
        2.0 * std::sqrt(ray_length_squared1 * ray_length_squared2);
    if (denominator == 0.0) {
        return 0.0;
    }
    const double nominator =
        ray_length_squared1 + ray_length_squared2 - baseline_length_squared;
    const double angle = std::abs(std::acos(nominator / denominator));

    // Triangulation is unstable for acute angles (far away points) and
    // obtuse angles (close points), so always compute the minimum angle
    // between the two intersecting rays.
    return std::min(angle, M_PI - angle);
}

std::vector<double> CalculateTriangulationAngles(
    const Eigen::Vector3d& proj_center1,
    const Eigen::Vector3d& proj_center2,
    const std::vector<Eigen::Vector3d>& points3D) {
    // Baseline length between camera centers.
    const double baseline_length_squared =
        (proj_center1 - proj_center2).squaredNorm();

    std::vector<double> angles(points3D.size());

    for (size_t i = 0; i < points3D.size(); ++i) {
        // Ray lengths from cameras to point.
        const double ray_length_squared1 =
            (points3D[i] - proj_center1).squaredNorm();
        const double ray_length_squared2 =
            (points3D[i] - proj_center2).squaredNorm();

        // Using "law of cosines" to compute the enclosing angle between rays.
        const double denominator =
            2.0 * std::sqrt(ray_length_squared1 * ray_length_squared2);
        if (denominator == 0.0) {
            angles[i] = 0.0;
            continue;
        }
        const double nominator =
            ray_length_squared1 + ray_length_squared2 - baseline_length_squared;
        const double angle = std::abs(std::acos(nominator / denominator));

        // Triangulation is unstable for acute angles (far away points) and
        // obtuse angles (close points), so always compute the minimum angle
        // between the two intersecting rays.
        angles[i] = std::min(angle, M_PI - angle);
    }

    return angles;
}