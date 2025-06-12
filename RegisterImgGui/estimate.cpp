#include <Eigen/Geometry>
#include <Eigen/LU>
#include <Eigen/SVD>
#include <Eigen/Eigenvalues>
#include "log.h"
#include "types.h"
#include "estimate.h"

int FindCubicPolynomialRoots(double c2,
    double c1,
    double c0,
    Eigen::Vector3d* real) {
    constexpr double k2PiOver3 = 2.09439510239319526263557236234192;
    constexpr double k4PiOver3 = 4.18879020478639052527114472468384;
    const double c2_over_3 = c2 / 3.0;
    const double a = c1 - c2 * c2_over_3;
    double b = (2.0 * c2 * c2 * c2 - 9.0 * c2 * c1) / 27.0 + c0;
    double c = b * b / 4.0 + a * a * a / 27.0;
    int num_roots = 0;
    if (c > 0) {
        c = std::sqrt(c);
        b *= -0.5;
        (*real)[0] = std::cbrt(b + c) + std::cbrt(b - c) - c2_over_3;
        num_roots = 1;
    }
    else {
        c = 3.0 * b / (2.0 * a) * std::sqrt(-3.0 / a);
        double d = 2.0 * std::sqrt(-a / 3.0);
        const double acos_over_3 = std::acos(c) / 3.0;
        (*real)[0] = d * std::cos(acos_over_3) - c2_over_3;
        (*real)[1] = d * std::cos(acos_over_3 - k2PiOver3) - c2_over_3;
        (*real)[2] = d * std::cos(acos_over_3 - k4PiOver3) - c2_over_3;
        num_roots = 3;
    }

    // Single Newton iteration.
    for (int i = 0; i < num_roots; ++i) {
        const double x = (*real)[i];
        const double x2 = x * x;
        const double x3 = x * x2;
        const double dx =
            -(x3 + c2 * x2 + c1 * x + c0) / (3 * x2 + 2 * c2 * x + c1);
        (*real)[i] += dx;
    }

    return num_roots;
}

bool ComputeSquaredSampsonError(const std::vector<Eigen::Vector2d>& points1,
    const std::vector<Eigen::Vector2d>& points2,
    const Eigen::Matrix3d& E,
    std::vector<double>* residuals) {
    const size_t num_points1 = points1.size();
    if (num_points1 != points2.size())
    {
        LOG_ERR_OUT << "num_points1!= points2.size()";
        return false;
    }
    residuals->resize(num_points1);
    for (size_t i = 0; i < num_points1; ++i) {
        const Eigen::Vector3d epipolar_line1 = E * points1[i].homogeneous();
        const Eigen::Vector3d point2_homogeneous = points2[i].homogeneous();
        const double num = point2_homogeneous.dot(epipolar_line1);
        const Eigen::Vector4d denom(point2_homogeneous.dot(E.col(0)),
            point2_homogeneous.dot(E.col(1)),
            epipolar_line1.x(),
            epipolar_line1.y());
        (*residuals)[i] = num * num / denom.squaredNorm();
    }
    return true;
}
bool CenterAndNormalizeImagePoints(const std::vector<Eigen::Vector2d>& points,
    std::vector<Eigen::Vector2d>* normed_points,
    Eigen::Matrix3d* normed_from_orig) {
    const size_t num_points = points.size();
    if (num_points==0)
    {
        LOG_ERR_OUT << "num_points==0";
        return false;
    }
    // Calculate centroid.
    Eigen::Vector2d centroid(0, 0);
    for (const Eigen::Vector2d& point : points) {
        centroid += point;
    }
    centroid /= num_points;

    // Root mean square distance to centroid of all points.
    double rms_mean_dist = 0;
    for (const Eigen::Vector2d& point : points) {
        rms_mean_dist += (point - centroid).squaredNorm();
    }
    rms_mean_dist = std::sqrt(rms_mean_dist / num_points);

    // Compose normalization matrix.
    const double norm_factor = std::sqrt(2.0) / rms_mean_dist;
    *normed_from_orig << norm_factor, 0, -norm_factor * centroid(0), 0,
        norm_factor, -norm_factor * centroid(1), 0, 0, 1;

    // Apply normalization matrix.
    normed_points->resize(num_points);
    for (size_t i = 0; i < num_points; ++i) {
        (*normed_points)[i] =
            (*normed_from_orig * points[i].homogeneous()).hnormalized();
    }
    return true;
}

bool FundamentalMatrixSevenPointEstimator::Estimate(
    const std::vector<X_t>& points1,
    const std::vector<Y_t>& points2,
    std::vector<M_t>* models) {
    if (points1.size() != 7 || points2.size() != 7)
    {
        LOG_ERR_OUT << "pts size!=7";
        return false;
    }
    if (models == nullptr)
    {
        LOG_ERR_OUT << "models == nullptr";
        return false;
    }

    models->clear();

    // Setup system of equations: [points2(i,:), 1]' * F * [points1(i,:), 1]'.
    Eigen::Matrix<double, 9, 7> A;
    for (size_t i = 0; i < 7; ++i) {
        A.col(i) << points1[i].x() * points2[i].homogeneous(),
            points1[i].y()* points2[i].homogeneous(), points2[i].homogeneous();
    }

    // 9 unknowns with 7 equations, so we have 2D null space.
    Eigen::Matrix<double, 9, 9> Q = A.fullPivHouseholderQr().matrixQ();

    // Normalize, such that lambda + mu = 1
    // and add constraint det(F) = det(lambda * f1 + (1 - lambda) * f2).

    auto f1 = Q.col(7);
    auto f2 = Q.col(8);
    f1 -= f2;

    const double t0 = f1(4) * f1(8) - f1(5) * f1(7);
    const double t1 = f1(3) * f1(8) - f1(5) * f1(6);
    const double t2 = f1(3) * f1(7) - f1(4) * f1(6);
    const double t3 = f2(4) * f2(8) - f2(5) * f2(7);
    const double t4 = f2(3) * f2(8) - f2(5) * f2(6);
    const double t5 = f2(3) * f2(7) - f2(4) * f2(6);

    Eigen::Vector4d coeffs;
    coeffs(0) = f1(0) * t0 - f1(1) * t1 + f1(2) * t2;
    if (std::abs(coeffs(0)) < 1e-16) {
        return false;
    }

    coeffs(1) = f2(0) * t0 - f2(1) * t1 + f2(2) * t2 -
        f2(3) * (f1(1) * f1(8) - f1(2) * f1(7)) +
        f2(4) * (f1(0) * f1(8) - f1(2) * f1(6)) -
        f2(5) * (f1(0) * f1(7) - f1(1) * f1(6)) +
        f2(6) * (f1(1) * f1(5) - f1(2) * f1(4)) -
        f2(7) * (f1(0) * f1(5) - f1(2) * f1(3)) +
        f2(8) * (f1(0) * f1(4) - f1(1) * f1(3));
    coeffs(2) = f1(0) * t3 - f1(1) * t4 + f1(2) * t5 -
        f1(3) * (f2(1) * f2(8) - f2(2) * f2(7)) +
        f1(4) * (f2(0) * f2(8) - f2(2) * f2(6)) -
        f1(5) * (f2(0) * f2(7) - f2(1) * f2(6)) +
        f1(6) * (f2(1) * f2(5) - f2(2) * f2(4)) -
        f1(7) * (f2(0) * f2(5) - f2(2) * f2(3)) +
        f1(8) * (f2(0) * f2(4) - f2(1) * f2(3));
    coeffs(3) = f2(0) * t3 - f2(1) * t4 + f2(2) * t5;

    coeffs.tail<3>() /= coeffs(0);

    Eigen::Vector3d roots;
    const int num_roots =
        FindCubicPolynomialRoots(coeffs(1), coeffs(2), coeffs(3), &roots);

    models->reserve(num_roots);
    for (int i = 0; i < num_roots; ++i) {
        const Eigen::Matrix<double, 9, 1> F = (f1 * roots[i] + f2).normalized();
        models->push_back(Eigen::Map<const Eigen::Matrix3d>(F.data()));
    }
}

void FundamentalMatrixSevenPointEstimator::Residuals(
    const std::vector<X_t>& points1,
    const std::vector<Y_t>& points2,
    const M_t& F,
    std::vector<double>* residuals)
{
    ComputeSquaredSampsonError(points1, points2, F, residuals);
}



bool FundamentalMatrixEightPointEstimator::Estimate(
    const std::vector<X_t>&points1,
    const std::vector<Y_t>&points2,
    std::vector<M_t>*models) 
{
    if (points1.size() <8 || points2.size() != points1.size())
    {
        LOG_ERR_OUT << "points1.size() <8 || points2.size() != points1.size()";
        return false;
    }
    if (models == nullptr)
    {
        LOG_ERR_OUT << "models == nullptr";
        return false;
    }
    models->clear();

    // Center and normalize image points for better numerical stability.
    std::vector<X_t> normed_points1;
    std::vector<Y_t> normed_points2;
    Eigen::Matrix3d normed_from_orig1;
    Eigen::Matrix3d normed_from_orig2;
    CenterAndNormalizeImagePoints(points1, &normed_points1, &normed_from_orig1);
    CenterAndNormalizeImagePoints(points2, &normed_points2, &normed_from_orig2);

    // Setup homogeneous linear equation as x2' * F * x1 = 0.
    Eigen::Matrix<double, Eigen::Dynamic, 9> A(points1.size(), 9);
    for (size_t i = 0; i < points1.size(); ++i) {
        A.row(i) << normed_points2[i].x() *
            normed_points1[i].transpose().homogeneous(),
            normed_points2[i].y()* normed_points1[i].transpose().homogeneous(),
            normed_points1[i].transpose().homogeneous();
    }

    // Solve for the nullspace of the constraint matrix.
    Eigen::Matrix3d Q;
    if (points1.size() == 8) {
        Eigen::Matrix<double, 9, 9> QQ =
            A.transpose().householderQr().householderQ();
        Q = Eigen::Map<const Eigen::Matrix<double, 3, 3, Eigen::RowMajor>>(
            QQ.col(8).data());
    }
    else {
        Eigen::JacobiSVD<Eigen::Matrix<double, Eigen::Dynamic, 9>> svd(
            A, Eigen::ComputeFullV);
        Q = Eigen::Map<const Eigen::Matrix<double, 3, 3, Eigen::RowMajor>>(
            svd.matrixV().col(8).data());
    }

    // Enforcing the internal constraint that two singular values must non-zero
    // and one must be zero.
    Eigen::JacobiSVD<Eigen::Matrix3d> svd(
        Q, Eigen::ComputeFullU | Eigen::ComputeFullV);
    Eigen::Vector3d singular_values = svd.singularValues();
    singular_values(2) = 0.0;
    const Eigen::Matrix3d F =
        svd.matrixU() * singular_values.asDiagonal() * svd.matrixV().transpose();

    models->resize(1);
    (*models)[0] = normed_from_orig2.transpose() * F * normed_from_orig1;
    return true;
}

void FundamentalMatrixEightPointEstimator::Residuals(
    const std::vector<X_t>&points1,
    const std::vector<Y_t>&points2,
    const M_t & E,
    std::vector<double>*residuals) {
    ComputeSquaredSampsonError(points1, points2, E, residuals);
}

InlierSupportMeasurer::Support InlierSupportMeasurer::Evaluate(
    const std::vector<double>& residuals, const double max_residual) {
    Support support;
    support.num_inliers = 0;
    support.residual_sum = 0;
    for (const auto residual : residuals) {
        if (residual <= max_residual) {
            support.num_inliers += 1;
            support.residual_sum += residual;
        }
    }
    return support;
}

bool InlierSupportMeasurer::IsLeftBetter(const Support& left,
    const Support& right) {
    if (left.num_inliers > right.num_inliers) {
        return true;
    }
    else {
        return left.num_inliers == right.num_inliers &&
            left.residual_sum < right.residual_sum;
    }
}

// Remove leading zero coefficients.
Eigen::VectorXd RemoveLeadingZeros(const Eigen::VectorXd& coeffs) 
{
    Eigen::VectorXd::Index num_zeros = 0;
    for (; num_zeros < coeffs.size(); ++num_zeros) {
        if (coeffs(num_zeros) != 0) {
            break;
        }
    }
    return coeffs.tail(coeffs.size() - num_zeros);
}
bool FindLinearPolynomialRoots(const Eigen::VectorXd& coeffs,
    Eigen::VectorXd* real,
    Eigen::VectorXd* imag) 
{
    if (coeffs.size() != 2)
    {
        LOG_ERR_OUT << "coeffs.size() != 2";
        return false;
    }
    if (coeffs(0) == 0) {
        return false;
    }
    if (real != nullptr) {
        real->resize(1);
        (*real)(0) = -coeffs(1) / coeffs(0);
    }
    if (imag != nullptr) {
        imag->resize(1);
        (*imag)(0) = 0;
    }
    return true;
}
// Remove trailing zero coefficients.
Eigen::VectorXd RemoveTrailingZeros(const Eigen::VectorXd& coeffs) 
{
    Eigen::VectorXd::Index num_zeros = 0;
    for (; num_zeros < coeffs.size(); ++num_zeros) {
        if (coeffs(coeffs.size() - 1 - num_zeros) != 0) {
            break;
        }
    }
    return coeffs.head(coeffs.size() - num_zeros);
}
bool FindQuadraticPolynomialRoots(const Eigen::VectorXd& coeffs,
    Eigen::VectorXd* real,
    Eigen::VectorXd* imag)
{
    if (coeffs.size() != 3)
    {
        LOG_ERR_OUT << "coeffs.size() != 3";
        return false;
    }
    const double a = coeffs(0);
    if (a == 0) {
        return FindLinearPolynomialRoots(coeffs.tail(2), real, imag);
    }
    const double b = coeffs(1);
    const double c = coeffs(2);
    if (b == 0 && c == 0) {
        if (real != nullptr) {
            real->resize(1);
            (*real)(0) = 0;
        }
        if (imag != nullptr) {
            imag->resize(1);
            (*imag)(0) = 0;
        }
        return true;
    }
    const double d = b * b - 4 * a * c;
    if (d >= 0) {
        const double sqrt_d = std::sqrt(d);
        if (real != nullptr) {
            real->resize(2);
            if (b >= 0) {
                (*real)(0) = (-b - sqrt_d) / (2 * a);
                (*real)(1) = (2 * c) / (-b - sqrt_d);
            }
            else {
                (*real)(0) = (2 * c) / (-b + sqrt_d);
                (*real)(1) = (-b + sqrt_d) / (2 * a);
            }
        }
        if (imag != nullptr) {
            imag->resize(2);
            imag->setZero();
        }
    }
    else {
        if (real != nullptr) {
            real->resize(2);
            real->setConstant(-b / (2 * a));
        }
        if (imag != nullptr) {
            imag->resize(2);
            (*imag)(0) = std::sqrt(-d) / (2 * a);
            (*imag)(1) = -(*imag)(0);
        }
    }
    return true;
}

bool FindPolynomialRootsCompanionMatrix(const Eigen::VectorXd& coeffs_all,
    Eigen::VectorXd* real,
    Eigen::VectorXd* imag) {
    if (coeffs_all.size() < 2)
    {
        LOG_ERR_OUT << "coeffs_all.size() < 2";
        return false;
    }
    Eigen::VectorXd coeffs = RemoveLeadingZeros(coeffs_all);
    const int degree = coeffs.size() - 1;
    if (degree <= 0) {
        return false;
    }
    else if (degree == 1) {
        return FindLinearPolynomialRoots(coeffs, real, imag);
    }
    else if (degree == 2) {
        return FindQuadraticPolynomialRoots(coeffs, real, imag);
    }
    // Remove the coefficients where zero is a solution.
    coeffs = RemoveTrailingZeros(coeffs);
    // Check if only zero is a solution.
    if (coeffs.size() == 1) {
        if (real != nullptr) {
            real->resize(1);
            (*real)(0) = 0;
        }
        if (imag != nullptr) {
            imag->resize(1);
            (*imag)(0) = 0;
        }
        return true;
    }
    // Fill the companion matrix.
    Eigen::MatrixXd C(coeffs.size() - 1, coeffs.size() - 1);
    C.setZero();
    for (Eigen::MatrixXd::Index i = 1; i < C.rows(); ++i) {
        C(i, i - 1) = 1;
    }
    C.row(0) = -coeffs.tail(coeffs.size() - 1) / coeffs(0);
    // Solve for the roots of the polynomial.
    Eigen::EigenSolver<Eigen::MatrixXd> solver(C, false);
    if (solver.info() != Eigen::Success) {
        return false;
    }

    // If there are trailing zeros, we must add zero as a solution.
    const int effective_degree =
        coeffs.size() - 1 < degree ? coeffs.size() : coeffs.size() - 1;

    if (real != nullptr) {
        real->resize(effective_degree);
        real->head(coeffs.size() - 1) = solver.eigenvalues().real();
        if (effective_degree > coeffs.size() - 1) {
            (*real)(real->size() - 1) = 0;
        }
    }
    if (imag != nullptr) {
        imag->resize(effective_degree);
        imag->head(coeffs.size() - 1) = solver.eigenvalues().imag();
        if (effective_degree > coeffs.size() - 1) {
            (*imag)(imag->size() - 1) = 0;
        }
    }

    return true;
}

bool EssentialMatrixFivePointEstimator::Estimate(
    const std::vector<X_t>& points1,
    const std::vector<Y_t>& points2,
    std::vector<M_t>* models) 
{
    if (points1.size() < 5 || points2.size() != points1.size())
    {
        LOG_ERR_OUT << "points1.size() <8 || points2.size() != points1.size()";
        return false;
    }
    if (models == nullptr)
    {
        LOG_ERR_OUT << "models == nullptr";
        return false;
    }
    models->clear();
    // Setup system of equations: [points2(i,:), 1]' * E * [points1(i,:), 1]'.
    Eigen::Matrix<double, Eigen::Dynamic, 9> Q(points1.size(), 9);
    for (size_t i = 0; i < points1.size(); ++i) {
        Q.row(i) << points2[i].x() * points1[i].transpose().homogeneous(),
            points2[i].y()* points1[i].transpose().homogeneous(),
            points1[i].transpose().homogeneous();
    }
    // Step 1: Extraction of the nullspace.
    Eigen::Matrix<double, 9, 4> E;
    if (points1.size() == 5) {
        E = Q.transpose().fullPivHouseholderQr().matrixQ().rightCols<4>();
    }
    else {
        const Eigen::JacobiSVD<Eigen::Matrix<double, Eigen::Dynamic, 9>> svd(
            Q, Eigen::ComputeFullV);
        E = svd.matrixV().rightCols<4>();
    }
    // Step 3: Gauss-Jordan elimination with partial pivoting on A.
    Eigen::Matrix<double, 10, 20> A;
    const Eigen::Matrix<double, 10, 10> AA =
        A.block<10, 10>(0, 0).partialPivLu().solve(A.block<10, 10>(0, 10));
    // Step 4: Expansion of the determinant polynomial of the 3x3 polynomial
    //         matrix B to obtain the tenth degree polynomial.
    Eigen::Matrix<double, 13, 3> B;
    for (size_t i = 0; i < 3; ++i) {
        B(0, i) = 0;
        B(4, i) = 0;
        B(8, i) = 0;
        B.block<3, 1>(1, i) = AA.block<1, 3>(i * 2 + 4, 0);
        B.block<3, 1>(5, i) = AA.block<1, 3>(i * 2 + 4, 3);
        B.block<4, 1>(9, i) = AA.block<1, 4>(i * 2 + 4, 6);
        B.block<3, 1>(0, i) -= AA.block<1, 3>(i * 2 + 5, 0);
        B.block<3, 1>(4, i) -= AA.block<1, 3>(i * 2 + 5, 3);
        B.block<4, 1>(8, i) -= AA.block<1, 4>(i * 2 + 5, 6);
    }
    // Step 5: Extraction of roots from the degree 10 polynomial.
    Eigen::Matrix<double, 11, 1> coeffs;
    Eigen::VectorXd roots_real;
    Eigen::VectorXd roots_imag;
    if (!FindPolynomialRootsCompanionMatrix(coeffs, &roots_real, &roots_imag)) {
        return;
    }
    const int num_roots = roots_real.size();
    models->reserve(num_roots);
    for (int i = 0; i < num_roots; ++i) {
        const double kMaxRootImag = 1e-10;
        if (std::abs(roots_imag(i)) > kMaxRootImag) {
            continue;
        }
        const double z1 = roots_real(i);
        const double z2 = z1 * z1;
        const double z3 = z2 * z1;
        const double z4 = z3 * z1;
        Eigen::Matrix3d Bz;
        for (int j = 0; j < 3; ++j) {
            Bz(j, 0) = B(0, j) * z3 + B(1, j) * z2 + B(2, j) * z1 + B(3, j);
            Bz(j, 1) = B(4, j) * z3 + B(5, j) * z2 + B(6, j) * z1 + B(7, j);
            Bz(j, 2) = B(8, j) * z4 + B(9, j) * z3 + B(10, j) * z2 + B(11, j) * z1 +
                B(12, j);
        }
        const Eigen::JacobiSVD<Eigen::Matrix3d> svd(Bz, Eigen::ComputeFullV);
        const Eigen::Vector3d X = svd.matrixV().rightCols<1>();
        const double kMaxX3 = 1e-10;
        if (std::abs(X(2)) < kMaxX3) {
            continue;
        }
        const Eigen::Matrix<double, 9, 1> e =
            (E.col(0) * (X(0) / X(2)) + E.col(1) * (X(1) / X(2)) + E.col(2) * z1 +
                E.col(3))
            .normalized();
        models->push_back(
            Eigen::Map<const Eigen::Matrix<double, 3, 3, Eigen::RowMajor>>(
                e.data()));
    }
    return true;
}

void EssentialMatrixFivePointEstimator::Residuals(
    const std::vector<X_t>& points1,
    const std::vector<Y_t>& points2,
    const M_t& E,
    std::vector<double>* residuals) {
    ComputeSquaredSampsonError(points1, points2, E, residuals);
}

bool EssentialMatrixEightPointEstimator::Estimate(
    const std::vector<X_t>& points1,
    const std::vector<Y_t>& points2,
    std::vector<M_t>* models) 
{
    if (points1.size() < 8 || points2.size() != points1.size())
    {
        LOG_ERR_OUT << "points1.size() <8 || points2.size() != points1.size()";
        return false;
    }
    if (models == nullptr)
    {
        LOG_ERR_OUT << "models == nullptr";
        return false;
    }
    models->clear();
    // Center and normalize image points for better numerical stability.
    std::vector<X_t> normed_points1;
    std::vector<Y_t> normed_points2;
    Eigen::Matrix3d normed_from_orig1;
    Eigen::Matrix3d normed_from_orig2;
    CenterAndNormalizeImagePoints(points1, &normed_points1, &normed_from_orig1);
    CenterAndNormalizeImagePoints(points2, &normed_points2, &normed_from_orig2);
    // Setup homogeneous linear equation as x2' * F * x1 = 0.
    Eigen::Matrix<double, Eigen::Dynamic, 9> A(points1.size(), 9);
    for (size_t i = 0; i < points1.size(); ++i) {
        A.row(i) << normed_points2[i].x() *
            normed_points1[i].transpose().homogeneous(),
            normed_points2[i].y()* normed_points1[i].transpose().homogeneous(),
            normed_points1[i].transpose().homogeneous();
    }
    // Solve for the nullspace of the constraint matrix.
    Eigen::Matrix3d Q;
    if (points1.size() == 8) {
        Eigen::Matrix<double, 9, 9> QQ =
            A.transpose().householderQr().householderQ();
        Q = Eigen::Map<const Eigen::Matrix<double, 3, 3, Eigen::RowMajor>>(
            QQ.col(8).data());
    }
    else {
        Eigen::JacobiSVD<Eigen::Matrix<double, Eigen::Dynamic, 9>> svd(
            A, Eigen::ComputeFullV);
        Q = Eigen::Map<const Eigen::Matrix<double, 3, 3, Eigen::RowMajor>>(
            svd.matrixV().col(8).data());
    }
    // Enforcing the internal constraint that two singular values must be non-zero
    // and one must be zero.
    Eigen::JacobiSVD<Eigen::Matrix3d> svd(
        Q, Eigen::ComputeFullU | Eigen::ComputeFullV);
    Eigen::Vector3d singular_values = svd.singularValues();
    singular_values(2) = 0.0;
    const Eigen::Matrix3d E =
        svd.matrixU() * singular_values.asDiagonal() * svd.matrixV().transpose();
    models->resize(1);
    (*models)[0] = normed_from_orig2.transpose() * E * normed_from_orig1;
    return true;
}

void EssentialMatrixEightPointEstimator::Residuals(
    const std::vector<X_t>& points1,
    const std::vector<Y_t>& points2,
    const M_t& E,
    std::vector<double>* residuals) {
    ComputeSquaredSampsonError(points1, points2, E, residuals);
}