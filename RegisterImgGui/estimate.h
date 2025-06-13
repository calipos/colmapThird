#ifndef _ESTIMATOR_H_
#define _ESTIMATOR_H_
#include <vector>
#include <random>
#include <algorithm>
#include <Eigen/Core>
#include "types.h"
#include "log.h"
#include "eigen_alignment.h"
// Fundamental matrix estimator from corresponding point pairs.
// This algorithm solves the 7-Point problem and is based on the following
// paper:
//    Zhengyou Zhang and T. Kanade, Determining the Epipolar Geometry and its
//    Uncertainty: A Review, International Journal of Computer Vision, 1998.
//    http://citeseerx.ist.psu.edu/viewdoc/summary?doi=10.1.1.33.4540
class FundamentalMatrixSevenPointEstimator {
public:
    typedef Eigen::Vector2d X_t;
    typedef Eigen::Vector2d Y_t;
    typedef Eigen::Matrix3d M_t;
    // The minimum number of samples needed to estimate a model.
    static const int kMinNumSamples = 7;
    // Estimate either 1 or 3 possible fundamental matrix solutions from a set of
    // corresponding points.
    // The number of corresponding points must be exactly 7.
    // @param points1  First set of corresponding points.
    // @param points2  Second set of corresponding points
    // @return         Up to 4 solutions as a vector of 3x3 fundamental matrices.
    static bool Estimate(const std::vector<X_t>& points1,
        const std::vector<Y_t>& points2,
        std::vector<M_t>* models);

    // Calculate the residuals of a set of corresponding points and a given
    // fundamental matrix.
    // Residuals are defined as the squared Sampson error.
    // @param points1    First set of corresponding points as Nx2 matrix.
    // @param points2    Second set of corresponding points as Nx2 matrix.
    // @param F          3x3 fundamental matrix.
    // @param residuals  Output vector of residuals.
    static void Residuals(const std::vector<X_t>& points1,
        const std::vector<Y_t>& points2,
        const M_t& F,
        std::vector<double>* residuals);
};

// Fundamental matrix estimator from corresponding point pairs.
// This algorithm solves the 8-Point problem based on the following paper:
//    Hartley and Zisserman, Multiple View Geometry, algorithm 11.1, page 282.
class FundamentalMatrixEightPointEstimator {
public:
    typedef Eigen::Vector2d X_t;
    typedef Eigen::Vector2d Y_t;
    typedef Eigen::Matrix3d M_t;
    // The minimum number of samples needed to estimate a model.
    static const int kMinNumSamples = 8;
    // Estimate fundamental matrix solutions from a set of corresponding points.
    // The number of corresponding points must be at least 8.
    // @param points1  First set of corresponding points.
    // @param points2  Second set of corresponding points
    // @return         Single solution as a vector of 3x3 fundamental matrices.
    static bool Estimate(const std::vector<X_t>& points1,
        const std::vector<Y_t>& points2,
        std::vector<M_t>* models);

    // Calculate the residuals of a set of corresponding points and a given
    // fundamental matrix.
    // Residuals are defined as the squared Sampson error.
    // @param points1    First set of corresponding points as Nx2 matrix.
    // @param points2    Second set of corresponding points as Nx2 matrix.
    // @param F          3x3 fundamental matrix.
    // @param residuals  Output vector of residuals.
    static void Residuals(const std::vector<X_t>& points1,
        const std::vector<Y_t>& points2,
        const M_t& F,
        std::vector<double>* residuals);
};


// Find the roots of a polynomial using the companion matrix method, based on:
//
//    R. A. Horn & C. R. Johnson, Matrix Analysis. Cambridge,
//    UK: Cambridge University Press, 1999, pp. 146-7.
//
// Compared to Durand-Kerner, this method is slower but more stable/accurate.
// The real and/or imaginary variable may be NULL if the output is not needed.
bool FindPolynomialRootsCompanionMatrix(const Eigen::VectorXd& coeffs,
    Eigen::VectorXd* real,
    Eigen::VectorXd* imag);

// Essential matrix estimator from corresponding normalized point pairs.
// This algorithm solves the 5-Point problem based on the following paper:
//    D. Nister, An efficient solution to the five-point relative pose problem,
//    IEEE-T-PAMI, 26(6), 2004.
//    http://citeseerx.ist.psu.edu/viewdoc/summary?doi=10.1.1.86.8769
class EssentialMatrixFivePointEstimator {
public:
    typedef Eigen::Vector2d X_t;
    typedef Eigen::Vector2d Y_t;
    typedef Eigen::Matrix3d M_t;
    // The minimum number of samples needed to estimate a model.
    static const int kMinNumSamples = 5;
    // Estimate up to 10 possible essential matrix solutions from a set of
    // corresponding points.
    //  The number of corresponding points must be at least 5.
    // @param points1  First set of corresponding points.
    // @param points2  Second set of corresponding points.
    // @return         Up to 10 solutions as a vector of 3x3 essential matrices.
    static bool Estimate(const std::vector<X_t>& points1,
        const std::vector<Y_t>& points2,
        std::vector<M_t>* models);
    // Calculate the residuals of a set of corresponding points and a given
    // essential matrix.
    // Residuals are defined as the squared Sampson error.
    // @param points1    First set of corresponding points.
    // @param points2    Second set of corresponding points.
    // @param E          3x3 essential matrix.
    // @param residuals  Output vector of residuals.
    static void Residuals(const std::vector<X_t>& points1,
        const std::vector<Y_t>& points2,
        const M_t& E,
        std::vector<double>* residuals);
};

// Essential matrix estimator from corresponding normalized point pairs.
// This algorithm solves the 8-Point problem based on the following paper:
//    Hartley and Zisserman, Multiple View Geometry, algorithm 11.1, page 282.
class EssentialMatrixEightPointEstimator {
public:
    typedef Eigen::Vector2d X_t;
    typedef Eigen::Vector2d Y_t;
    typedef Eigen::Matrix3d M_t;
    // The minimum number of samples needed to estimate a model.
    static const int kMinNumSamples = 8;
    // Estimate essential matrix solutions from  set of corresponding points.
    // The number of corresponding points must be at least 8.
    // @param points1  First set of corresponding points.
    // @param points2  Second set of corresponding points.
    static bool Estimate(const std::vector<X_t>& points1,
        const std::vector<Y_t>& points2,
        std::vector<M_t>* models);
    // Calculate the residuals of a set of corresponding points and a given
    // essential matrix.
    // Residuals are defined as the squared Sampson error.
    // @param points1    First set of corresponding points.
    // @param points2    Second set of corresponding points.
    // @param E          3x3 essential matrix.
    // @param residuals  Output vector of residuals.
    static void Residuals(const std::vector<X_t>& points1,
        const std::vector<Y_t>& points2,
        const M_t& E,
        std::vector<double>* residuals);
};



// Measure the support of a model by counting the number of inliers and
// summing all inlier residuals. The support is better if it has more inliers
// and a smaller residual sum.
struct InlierSupportMeasurer {
    struct Support {
        // The number of inliers.
        size_t num_inliers = 0;
        // The sum of all inlier residuals.
        double residual_sum = std::numeric_limits<double>::max();
    };
    // Compute the support of the residuals.
    Support Evaluate(const std::vector<double>& residuals, double max_residual);
    // Compare the two supports.
    bool IsLeftBetter(const Support& left, const Support& right);
};

template <typename Estimator,
    typename SupportMeasurer = InlierSupportMeasurer>
struct Report {
    // Whether the estimation was successful.
    bool success = false;
    // The number of RANSAC trials / iterations.
    size_t num_trials = 0;
    // The support of the estimated model.
    typename SupportMeasurer::Support support;
    // Boolean mask which is true if a sample is an inlier.
    std::vector<char> inlier_mask;
    // The estimated model.
    typename Estimator::M_t model;
};


//---------------------------------------------------------------------------
struct RANSACOptions {
    // Maximum error for a sample to be considered as an inlier. Note that
    // the residual of an estimator corresponds to a squared error.
    double max_error = 0.0;
    // A priori assumed minimum inlier ratio, which determines the maximum number
    // of iterations. Only applies if smaller than `max_num_trials`.
    double min_inlier_ratio = 0.1;
    // Abort the iteration if minimum probability that one sample is free from
    // outliers is reached.
    double confidence = 0.99;
    // The num_trials_multiplier to the dynamically computed maximum number of
    // iterations based on the specified confidence value.
    double dyn_num_trials_multiplier = 3.0;
    // Number of random trials to estimate model from random subset.
    int min_num_trials = 0;
    int max_num_trials = std::numeric_limits<int>::max();
};
template <typename Estimator,
typename SupportMeasurer = InlierSupportMeasurer>
class RANSAC {
public:
    struct Report {
        // Whether the estimation was successful.
        bool success = false;
        // The number of RANSAC trials / iterations.
        size_t num_trials = 0;
        // The support of the estimated model.
        typename SupportMeasurer::Support support;
        // Boolean mask which is true if a sample is an inlier.
        std::vector<char> inlier_mask;
        // The estimated model.
        typename Estimator::M_t model;
    };
    explicit RANSAC(const RANSACOptions& options);
    // Robustly estimate model with RANSAC (RANdom SAmple Consensus).
    // @param X              Independent variables.
    // @param Y              Dependent variables.
    // @return               The report with the results of the estimation.
    Report Estimate(
        const std::vector< typename Estimator::X_t>& featPts1,
        const std::vector< typename Estimator::Y_t>& featPts2);
    // Objects used in RANSAC procedure. Access useful to define custom behavior
    // through options or e.g. to compute residuals.
    Estimator estimator;
    SupportMeasurer support_measurer;
protected:
    RANSACOptions options_;
};

template <typename Estimator, typename SupportMeasurer>
RANSAC<Estimator, SupportMeasurer>::RANSAC(
    const RANSACOptions& options)
    : options_(options) {
    // Determine max_num_trials based on assumed `min_inlier_ratio`.
    const size_t kNumSamples = 10;
}
template <typename Estimator, typename SupportMeasurer>
typename RANSAC<Estimator, SupportMeasurer>::Report
RANSAC<Estimator, SupportMeasurer>::Estimate(
    const std::vector< typename Estimator::X_t>& featPts1,
    const std::vector< typename Estimator::Y_t>& featPts2) 
{
    Report report;
    report.success = false;
    report.num_trials = 0;
    return report;
}


// Implementation of LO-RANSAC (Locally Optimized RANSAC).
// "Locally Optimized RANSAC" Ondrej Chum, Jiri Matas, Josef Kittler, DAGM 2003.
template <typename Estimator, typename LocalEstimator, typename SupportMeasurer = InlierSupportMeasurer>
class LORANSAC : public RANSAC<Estimator, SupportMeasurer>
{
public:
    using typename RANSAC<Estimator, SupportMeasurer>::Report;
    explicit LORANSAC(const RANSACOptions& options);
    // Robustly estimate model with RANSAC (RANdom SAmple Consensus).
    // @param X              Independent variables.
    // @param Y              Dependent variables.
    // @return               The report with the results of the estimation.
    Report Estimate(
        const std::vector< typename Estimator::X_t>& featPts1,
        const std::vector< typename Estimator::Y_t>& featPts2);
    int Combination1(const int& n, const  const int& m);
    using RANSAC<Estimator, SupportMeasurer>::estimator;
    LocalEstimator local_estimator;
    using RANSAC<Estimator, SupportMeasurer>::support_measurer;
private:
        using RANSAC<Estimator, SupportMeasurer>::options_;
};

template <typename Estimator,
    typename LocalEstimator,
    typename SupportMeasurer>
    LORANSAC<Estimator, LocalEstimator, SupportMeasurer>::LORANSAC(const RANSACOptions& options)
    : RANSAC<Estimator, SupportMeasurer>(options) {}
template <typename Estimator, typename LocalEstimator, typename SupportMeasurer>
int LORANSAC<Estimator, LocalEstimator, SupportMeasurer>::Combination1(const int& n, const  const int& m) 
{
    if (n <= 0 || m <= 0)
    {
        return 0;
    }
    int res = 1;
    for (int i = 1; i <= m; ++i) {
        res = res * (n - m + i) / i;        // ÏÈ³Ëºó³ý
    }
    return res;
}
template <typename Estimator, typename LocalEstimator, typename SupportMeasurer>
typename LORANSAC<Estimator, LocalEstimator, SupportMeasurer>::Report LORANSAC<Estimator, LocalEstimator, SupportMeasurer>::Estimate(
    const std::vector<typename Estimator::X_t>& featPts1,
    const std::vector<typename Estimator::Y_t>& featPts2)
{
    typename RANSAC<Estimator, SupportMeasurer>::Report report;
    report.success = false;
    report.num_trials = 0;
    if (featPts1.size()!= featPts2.size())
    {
        LOG_ERR_OUT << "featPts1.size()!= featPts2.size()";
        return report;
    }

    const size_t num_samples = featPts1.size();
    std::vector<size_t>matches(num_samples);
    if (num_samples < Estimator::kMinNumSamples) 
    {
        return report;
    }
    std::vector<Eigen::Vector2d> X(num_samples);
    std::vector<Eigen::Vector2d> Y(num_samples);
    for (size_t i = 0; i < num_samples; ++i) {
        X[i] = featPts1[i];
        Y[i] = featPts2[i];
        matches[i] = i;
    }
    typename SupportMeasurer::Support best_support;
    typename Estimator::M_t best_model;
    bool best_model_is_local = false;
    const double max_residual = options_.max_error * options_.max_error;

    std::vector<double> residuals;
    std::vector<double> best_local_residuals;

    std::vector<typename LocalEstimator::X_t> X_inlier;
    std::vector<typename LocalEstimator::Y_t> Y_inlier;

    std::vector<typename Estimator::X_t> X_rand(Estimator::kMinNumSamples);
    std::vector<typename Estimator::Y_t> Y_rand(Estimator::kMinNumSamples);
    std::vector<typename Estimator::M_t> sample_models;
    std::vector<typename LocalEstimator::M_t> local_models;
    

    int maxTrialsNum = std::min(Combination1(num_samples, Estimator::kMinNumSamples), 200);
    for (report.num_trials = 0; report.num_trials < maxTrialsNum;
        ++report.num_trials) 
    {
        report.num_trials += 1;
        std::shuffle(matches.begin(), matches.end(), std::default_random_engine(std::time(0)));
        for (size_t i = 0; i < Estimator::kMinNumSamples; ++i) {
            X_rand[i] = featPts1[matches[i]];
            Y_rand[i] = featPts2[matches[i]];
        }
        // Estimate model for current subset.
        estimator.Estimate(X_rand, Y_rand, &sample_models);

        // Iterate through all estimated models
        for (const auto& sample_model : sample_models) {
            estimator.Residuals(X, Y, sample_model, &residuals);
            const auto support = support_measurer.Evaluate(residuals, max_residual);
            // Do local optimization if better than all previous subsets.
            if (support_measurer.IsLeftBetter(support, best_support)) {
                best_support = support;
                best_model = sample_model;
                best_model_is_local = false;
                // Estimate locally optimized model from inliers.
                if (support.num_inliers > Estimator::kMinNumSamples &&
                    support.num_inliers >= LocalEstimator::kMinNumSamples)
                {
                    X_inlier.clear();
                    Y_inlier.clear();
                    X_inlier.reserve(num_samples);
                    Y_inlier.reserve(num_samples);
                    for (size_t i = 0; i < residuals.size(); ++i) {
                        if (residuals[i] <= max_residual) {
                            X_inlier.push_back(X[i]);
                            Y_inlier.push_back(Y[i]);
                        }
                    }
                    local_estimator.Estimate(X_inlier, Y_inlier, &local_models);
                    const size_t prev_best_num_inliers = best_support.num_inliers;
                    for (const auto& local_model : local_models) {
                        local_estimator.Residuals(X, Y, local_model, &residuals);
                        const auto local_support =
                            support_measurer.Evaluate(residuals, max_residual);
                        // Check if locally optimized model is better.
                        if (support_measurer.IsLeftBetter(local_support, best_support)) {
                            best_support = local_support;
                            best_model = local_model;
                            best_model_is_local = true;
                            std::swap(residuals, best_local_residuals);
                        }
                    }

                    // Only continue recursive local optimization, if the inlier set
                    // size increased and we thus have a chance to further improve.
                    //if (best_support.num_inliers <= prev_best_num_inliers) {
                    //    break;
                    //}

                    // Swap back the residuals, so we can extract the best inlier
                    // set in the next recursion of local optimization.
                    std::swap(residuals, best_local_residuals);

                }
            }
        }
    }

    report.support = best_support;
    report.model = best_model;

    // No valid model was found
    if (report.support.num_inliers < estimator.kMinNumSamples) {
        return report;
    }

    report.success = true;

    // Determine inlier mask. Note that this calculates the residuals for the
    // best model twice, but saves to copy and fill the inlier mask for each
    // evaluated model. Some benchmarking revealed that this approach is faster.

    if (best_model_is_local) {
        local_estimator.Residuals(X, Y, report.model, &residuals);
    }
    else {
        estimator.Residuals(X, Y, report.model, &residuals);
    }
    report.inlier_mask.resize(num_samples);
    for (size_t i = 0; i < residuals.size(); ++i) {
        report.inlier_mask[i] = residuals[i] <= max_residual;
    }
    return report;
}

#endif // !_ESTIMATOR_H_