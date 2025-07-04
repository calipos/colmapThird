#ifndef _BUNDLE_ADJUSTMENT_H_
#define _BUNDLE_ADJUSTMENT_H_


#include <memory>
#include <unordered_set>
#include <Eigen/Core>
#include "ceres/ceres.h"
#include "types.h"
#include "reconstruction.h"
#include "camera_rig.h"
#include "image.h"
// Configuration container to setup bundle adjustment problems.
class BundleAdjustmentConfig {
public:
    BundleAdjustmentConfig();

    size_t NumImages() const;
    size_t NumPoints() const;
    size_t NumConstantCamIntrinsics() const;
    size_t NumConstantCamPoses() const;
    size_t NumConstantCamPositions() const;
    size_t NumVariablePoints() const;
    size_t NumConstantPoints() const;

    // Determine the number of residuals for the given reconstruction. The number
    // of residuals equals the number of observations times two.
    size_t NumResiduals(const std::vector<class Image>& imageList) const;

    // Add / remove images from the configuration.
    void AddImage(image_t image_id);
    bool HasImage(image_t image_id) const;
    void RemoveImage(image_t image_id);

    // Set cameras of added images as constant or variable. By default all
    // cameras of added images are variable. Note that the corresponding images
    // have to be added prior to calling these methods.
    void SetConstantCamIntrinsics(camera_t camera_id);
    void SetVariableCamIntrinsics(camera_t camera_id);
    bool HasConstantCamIntrinsics(camera_t camera_id) const;

    // Set the pose of added images as constant. The pose is defined as the
    // rotational and translational part of the projection matrix.
    void SetConstantCamPose(image_t image_id);
    void SetVariableCamPose(image_t image_id);
    bool HasConstantCamPose(image_t image_id) const;

    // Set the translational part of the pose, hence the constant pose
    // indices may be in [0, 1, 2] and must be unique. Note that the
    // corresponding images have to be added prior to calling these methods.
    void SetConstantCamPositions(image_t image_id, const std::vector<int>& idxs);
    void RemoveConstantCamPositions(image_t image_id);
    bool HasConstantCamPositions(image_t image_id) const;

    // Add / remove points from the configuration. Note that points can either
    // be variable or constant but not both at the same time.
    void AddVariablePoint(point3D_t point3D_id);
    void AddConstantPoint(point3D_t point3D_id);
    bool HasPoint(point3D_t point3D_id) const;
    bool HasVariablePoint(point3D_t point3D_id) const;
    bool HasConstantPoint(point3D_t point3D_id) const;
    void RemoveVariablePoint(point3D_t point3D_id);
    void RemoveConstantPoint(point3D_t point3D_id);

    // Access configuration data.
    const std::unordered_set<camera_t> ConstantIntrinsics() const;
    const std::unordered_set<image_t>& Images() const;
    const std::unordered_set<point3D_t>& VariablePoints() const;
    const std::unordered_set<point3D_t>& ConstantPoints() const;
    const std::unordered_set<image_t>& ConstantCamPoses() const;
    const std::vector<int>& ConstantCamPositions(image_t image_id) const;

private:
    std::unordered_set<camera_t> constant_intrinsics_;
    std::unordered_set<image_t> image_ids_;
    std::unordered_set<point3D_t> variable_point3D_ids_;
    std::unordered_set<point3D_t> constant_point3D_ids_;
    std::unordered_set<image_t> constant_cam_poses_;
    std::unordered_map<image_t, std::vector<int>> constant_cam_positions_;
};

struct BundleAdjustmentOptions {
    // Loss function types: Trivial (non-robust) and Cauchy (robust) loss.
    enum class LossFunctionType { TRIVIAL, SOFT_L1, CAUCHY };
    LossFunctionType loss_function_type = LossFunctionType::TRIVIAL;

    // Scaling factor determines residual at which robustification takes place.
    double loss_function_scale = 1.0;

    // Whether to refine the focal length parameter group.
    bool refine_focal_length = true;

    // Whether to refine the principal point parameter group.
    bool refine_principal_point = false;

    // Whether to refine the extra parameter group.
    bool refine_extra_params = true;

    // Whether to refine the extrinsic parameter group.
    bool refine_extrinsics = true;

    // Whether to print a final summary.
    bool print_summary = true;

    // Whether to use Ceres' CUDA linear algebra library, if available.
    bool use_gpu = false;
    std::string gpu_index = "-1";

    // Heuristic threshold to switch from CPU to GPU based solvers.
    // Typically, the GPU is faster for large problems but the overhead of
    // transferring memory from the CPU to the GPU leads to better CPU performance
    // for small problems. This depends on the specific problem and hardware.
    int min_num_images_gpu_solver = 50;

    // Heuristic threshold on the minimum number of residuals to enable
    // multi-threading. Note that single-threaded is typically better for small
    // bundle adjustment problems due to the overhead of threading.
    int min_num_residuals_for_cpu_multi_threading = 50000;

    // Heuristic thresholds to switch between direct, sparse, and iterative
    // solvers. These thresholds may not be optimal for all types of problems.
    int max_num_images_direct_dense_cpu_solver = 50;
    int max_num_images_direct_sparse_cpu_solver = 1000;
    int max_num_images_direct_dense_gpu_solver = 200;
    int max_num_images_direct_sparse_gpu_solver = 4000;

    // Ceres-Solver options.
    ceres::Solver::Options solver_options;

    BundleAdjustmentOptions() {
        solver_options.function_tolerance = 0.0;
        solver_options.gradient_tolerance = 1e-4;
        solver_options.parameter_tolerance = 0.0;
        solver_options.logging_type = ceres::LoggingType::SILENT;
        solver_options.max_num_iterations = 100;
        solver_options.max_linear_solver_iterations = 200;
        solver_options.max_num_consecutive_invalid_steps = 10;
        solver_options.max_consecutive_nonmonotonic_steps = 10;
        solver_options.num_threads = -1;
#if CERES_VERSION_MAJOR < 2
        solver_options.num_linear_solver_threads = -1;
#endif  // CERES_VERSION_MAJOR
    }

    // Create a new loss function based on the specified options. The caller
    // takes ownership of the loss function.
    ceres::LossFunction* CreateLossFunction() const;

    // Create options tailored for given bundle adjustment config and problem.
    ceres::Solver::Options CreateSolverOptions(
        const BundleAdjustmentConfig& config,
        const ceres::Problem& problem) const;

    bool Check() const;
};

struct RigBundleAdjustmentOptions {
    // Whether to optimize the relative poses of the camera rigs.
    bool refine_relative_poses = true;

    // The maximum allowed reprojection error for an observation to be
    // considered in the bundle adjustment. Some observations might have large
    // reprojection errors due to the concatenation of the absolute and relative
    // rig poses, which might be different from the absolute pose of the image
    // in the reconstruction.
    double max_reproj_error = 1000.0;
};

struct PosePriorBundleAdjustmentOptions {
    // Whether to use a robust loss on prior locations.
    bool use_robust_loss_on_prior_position = false;

    // Threshold on the residual for the robust loss
    // (chi2 for 3DOF at 95% = 7.815).
    double prior_position_loss_scale = 7.815;

    // Maximum RANSAC error for Sim3 alignment.
    double ransac_max_error = 0.;
};

class BundleAdjuster {
public:
    BundleAdjuster(BundleAdjustmentOptions options,
        BundleAdjustmentConfig config);
    virtual ~BundleAdjuster() = default;

    virtual ceres::Solver::Summary Solve() = 0;
    virtual std::shared_ptr<ceres::Problem>& Problem() = 0;

    const BundleAdjustmentOptions& Options() const;
    const BundleAdjustmentConfig& Config() const;

protected:
    BundleAdjustmentOptions options_;
    BundleAdjustmentConfig config_;
};

std::unique_ptr<BundleAdjuster> CreateDefaultBundleAdjuster(
    BundleAdjustmentOptions options,
    BundleAdjustmentConfig config,
    std::vector<struct Camera>&cameraList,
    std::vector<class Image> &imageList,
    std::unordered_map<point3D_t, Eigen::Vector3d>&objPts);

std::unique_ptr<BundleAdjuster> CreateRigBundleAdjuster(
    BundleAdjustmentOptions options,
    RigBundleAdjustmentOptions rig_options,
    BundleAdjustmentConfig config,
    std::vector<struct Camera>& cameraList,
    std::vector<class Image>& imageList,
    std::unordered_map<point3D_t, Eigen::Vector3d>& objPts,
    std::vector<CameraRig>& camera_rigs);

std::unique_ptr<BundleAdjuster> CreatePosePriorBundleAdjuster(
    BundleAdjustmentOptions options,
    PosePriorBundleAdjustmentOptions prior_options,
    BundleAdjustmentConfig config,
    //std::unordered_map<image_t, PosePrior> pose_priors,
    std::vector<struct Camera>& cameraList,
    std::vector<class Image>& imageList,
    std::unordered_map<point3D_t, Eigen::Vector3d>& objPts);

void PrintSolverSummary(const ceres::Solver::Summary& summary,
    const std::string& header);


#endif // !_BUNDLE_ADJUSTMENT_H_
