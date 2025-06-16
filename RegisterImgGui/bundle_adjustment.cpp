
#include "bundle_adjustment.h"

//#include "alignment.h"
#include "cost_functions.h"
#include "manifold.h"
#include "projection.h"
#include "camera.h"
#include "misc.h"
#include "threading.h"
#include "timer.h"
#include <unordered_set>
#include <iomanip>
#include <unordered_map>


////////////////////////////////////////////////////////////////////////////////
// BundleAdjustmentConfig
////////////////////////////////////////////////////////////////////////////////

BundleAdjustmentConfig::BundleAdjustmentConfig() {}

size_t BundleAdjustmentConfig::NumImages() const { return image_ids_.size(); }

size_t BundleAdjustmentConfig::NumPoints() const {
    return variable_point3D_ids_.size() + constant_point3D_ids_.size();
}

size_t BundleAdjustmentConfig::NumConstantCamIntrinsics() const {
    return constant_intrinsics_.size();
}

size_t BundleAdjustmentConfig::NumConstantCamPoses() const {
    return constant_cam_poses_.size();
}

size_t BundleAdjustmentConfig::NumConstantCamPositions() const {
    return constant_cam_positions_.size();
}

size_t BundleAdjustmentConfig::NumVariablePoints() const {
    return variable_point3D_ids_.size();
}

size_t BundleAdjustmentConfig::NumConstantPoints() const {
    return constant_point3D_ids_.size();
}

size_t BundleAdjustmentConfig::NumResiduals(const std::vector<class Image>&imageList) const {
    std::unordered_set<point2D_t>featIds;
    for (const auto& imageId : image_ids_)
    {
        for (const auto&d: imageList[imageId].featPts)
        {
            featIds.insert(d.first);
        }
    }  
    return 2 * featIds.size();
}

void BundleAdjustmentConfig::AddImage(const image_t image_id) {
    image_ids_.insert(image_id);
}

bool BundleAdjustmentConfig::HasImage(const image_t image_id) const {
    return image_ids_.find(image_id) != image_ids_.end();
}

void BundleAdjustmentConfig::RemoveImage(const image_t image_id) {
    image_ids_.erase(image_id);
}

void BundleAdjustmentConfig::SetConstantCamIntrinsics(
    const camera_t camera_id) {
    constant_intrinsics_.insert(camera_id);
}

void BundleAdjustmentConfig::SetVariableCamIntrinsics(
    const camera_t camera_id) {
    constant_intrinsics_.erase(camera_id);
}

bool BundleAdjustmentConfig::HasConstantCamIntrinsics(
    const camera_t camera_id) const {
    return constant_intrinsics_.find(camera_id) != constant_intrinsics_.end();
}

void BundleAdjustmentConfig::SetConstantCamPose(const image_t image_id) {
    if (!HasImage(image_id))
    {
        LOG_ERR_OUT << "not HasImage : " << image_id;
        return;
    }
    if (HasConstantCamPositions(image_id))
    {
        LOG_ERR_OUT << "HasConstantCamPositions : " << image_id;
        return;
    }
    constant_cam_poses_.insert(image_id);
}

void BundleAdjustmentConfig::SetVariableCamPose(const image_t image_id) {
    constant_cam_poses_.erase(image_id);
}

bool BundleAdjustmentConfig::HasConstantCamPose(const image_t image_id) const {
    return constant_cam_poses_.find(image_id) != constant_cam_poses_.end();
}

void BundleAdjustmentConfig::SetConstantCamPositions(
    const image_t image_id, const std::vector<int>& idxs) {
    if (idxs.size() == 0)
    {
        LOG_ERR_OUT << "idxs.size()==0";
        return;
    }
    if (idxs.size() >= 3)
    {
        LOG_ERR_OUT << "idxs.size() < 3";
        return;
    }
    if (!HasImage(image_id))
    {
        LOG_ERR_OUT << "not HasImage : " << image_id;
        return;
    }
    if (HasConstantCamPose(image_id))
    {
        LOG_ERR_OUT << "HasConstantCamPose : " << image_id;
        return;
    }
    if (VectorContainsDuplicateValues(idxs))
    {
        LOG_ERR_OUT << "VectorContainsDuplicateValues : idxs";
        return;
    }
    constant_cam_positions_.emplace(image_id, idxs);
}

void BundleAdjustmentConfig::RemoveConstantCamPositions(
    const image_t image_id) {
    constant_cam_positions_.erase(image_id);
}

bool BundleAdjustmentConfig::HasConstantCamPositions(
    const image_t image_id) const {
    return constant_cam_positions_.find(image_id) !=
        constant_cam_positions_.end();
}

const std::unordered_set<camera_t> BundleAdjustmentConfig::ConstantIntrinsics()
    const {
    return constant_intrinsics_;
}

const std::unordered_set<image_t>& BundleAdjustmentConfig::Images() const {
    return image_ids_;
}

const std::unordered_set<point3D_t>& BundleAdjustmentConfig::VariablePoints()
    const {
    return variable_point3D_ids_;
}

const std::unordered_set<point3D_t>& BundleAdjustmentConfig::ConstantPoints()
    const {
    return constant_point3D_ids_;
}

const std::unordered_set<image_t>& BundleAdjustmentConfig::ConstantCamPoses()
    const {
    return constant_cam_poses_;
}

const std::vector<int>& BundleAdjustmentConfig::ConstantCamPositions(
    const image_t image_id) const {
    return constant_cam_positions_.at(image_id);
}

void BundleAdjustmentConfig::AddVariablePoint(const point3D_t point3D_id) {
    if (HasConstantPoint(point3D_id))
    {
        LOG_ERR_OUT << "HasConstantPoint : "<< point3D_id;
        return;
    }
    variable_point3D_ids_.insert(point3D_id);
}

void BundleAdjustmentConfig::AddConstantPoint(const point3D_t point3D_id) {
    if (HasVariablePoint(point3D_id))
    {
        LOG_ERR_OUT << "HasVariablePoint : " << point3D_id;
        return;
    }
    constant_point3D_ids_.insert(point3D_id);
}

bool BundleAdjustmentConfig::HasPoint(const point3D_t point3D_id) const {
    return HasVariablePoint(point3D_id) || HasConstantPoint(point3D_id);
}

bool BundleAdjustmentConfig::HasVariablePoint(
    const point3D_t point3D_id) const {
    return variable_point3D_ids_.find(point3D_id) != variable_point3D_ids_.end();
}

bool BundleAdjustmentConfig::HasConstantPoint(
    const point3D_t point3D_id) const {
    return constant_point3D_ids_.find(point3D_id) != constant_point3D_ids_.end();
}

void BundleAdjustmentConfig::RemoveVariablePoint(const point3D_t point3D_id) {
    variable_point3D_ids_.erase(point3D_id);
}

void BundleAdjustmentConfig::RemoveConstantPoint(const point3D_t point3D_id) {
    constant_point3D_ids_.erase(point3D_id);
}

BundleAdjuster::BundleAdjuster(BundleAdjustmentOptions options,
    BundleAdjustmentConfig config)
    : options_(std::move(options)), config_(std::move(config)) {
    if (!options_.Check())
    {
        LOG_ERR_OUT << "options_.Check()";
        return;
    }
}

const BundleAdjustmentOptions& BundleAdjuster::Options() const {
    return options_;
}

const BundleAdjustmentConfig& BundleAdjuster::Config() const { return config_; }

////////////////////////////////////////////////////////////////////////////////
// BundleAdjustmentOptions
////////////////////////////////////////////////////////////////////////////////

ceres::LossFunction* BundleAdjustmentOptions::CreateLossFunction() const {
    ceres::LossFunction* loss_function = nullptr;
    switch (loss_function_type) {
    case LossFunctionType::TRIVIAL:
        loss_function = new ceres::TrivialLoss();
        break;
    case LossFunctionType::SOFT_L1:
        loss_function = new ceres::SoftLOneLoss(loss_function_scale);
        break;
    case LossFunctionType::CAUCHY:
        loss_function = new ceres::CauchyLoss(loss_function_scale);
        break;
    }
    if (loss_function==nullptr)
    {
        LOG_ERR_OUT << "loss_function==nullptr";
    }
    return loss_function;
}

ceres::Solver::Options BundleAdjustmentOptions::CreateSolverOptions(
    const BundleAdjustmentConfig& config, const ceres::Problem& problem) const {
    ceres::Solver::Options custom_solver_options = solver_options;
    if (VLOG_IS_ON(2)) {
        custom_solver_options.minimizer_progress_to_stdout = true;
        custom_solver_options.logging_type =
            ceres::LoggingType::PER_MINIMIZER_ITERATION;
    }

    const int num_images = config.NumImages();
    const bool has_sparse =
        custom_solver_options.sparse_linear_algebra_library_type !=
        ceres::NO_SPARSE;

    int max_num_images_direct_dense_solver =
        max_num_images_direct_dense_cpu_solver;
    int max_num_images_direct_sparse_solver =
        max_num_images_direct_sparse_cpu_solver;

    if (num_images <= max_num_images_direct_dense_solver) {
        custom_solver_options.linear_solver_type = ceres::DENSE_SCHUR;
    }
    else if (has_sparse && num_images <= max_num_images_direct_sparse_solver) {
        custom_solver_options.linear_solver_type = ceres::SPARSE_SCHUR;
    }
    else {  // Indirect sparse (preconditioned CG) solver.
        custom_solver_options.linear_solver_type = ceres::ITERATIVE_SCHUR;
        custom_solver_options.preconditioner_type = ceres::SCHUR_JACOBI;
    }

    if (problem.NumResiduals() < min_num_residuals_for_cpu_multi_threading) {
        custom_solver_options.num_threads = 1;
#if CERES_VERSION_MAJOR < 2
        custom_solver_options.num_linear_solver_threads = 1;
#endif  // CERES_VERSION_MAJOR
    }
    else {
        custom_solver_options.num_threads =
            GetEffectiveNumThreads(custom_solver_options.num_threads);
#if CERES_VERSION_MAJOR < 2
        custom_solver_options.num_linear_solver_threads =
            GetEffectiveNumThreads(custom_solver_options.num_linear_solver_threads);
#endif  // CERES_VERSION_MAJOR
    }

    std::string solver_error;
    if (!custom_solver_options.IsValid(&solver_error))
    {
        LOG_ERR_OUT << solver_error;
    }
    return custom_solver_options;
}

bool BundleAdjustmentOptions::Check() const {
    if (loss_function_scale<0)
    {
        LOG_ERR_OUT << "loss_function_scale<0";
        return false;
    }
    if (max_num_images_direct_dense_cpu_solver >= max_num_images_direct_sparse_cpu_solver)
    {
        LOG_ERR_OUT << "max_num_images_direct_dense_cpu_solver >= max_num_images_direct_sparse_cpu_solver";
        return false;
    }
    return true;
}

namespace {

    void ParameterizeCameras(const BundleAdjustmentOptions& options,
        const BundleAdjustmentConfig& config,
        std::vector<struct Camera>& cameraList,
        std::vector<class Image>& imageList,
        std::unordered_map<point3D_t, Eigen::Vector3d>& objPts,
        ceres::Problem& problem) {
        const bool constant_camera = !options.refine_focal_length &&
            !options.refine_principal_point &&
            !options.refine_extra_params;
        for (size_t i = 0; i < cameraList.size(); i++)
        {
            Camera& camera = cameraList[i];
            camera_t camera_id = static_cast<camera_t>(i);
            if (constant_camera || config.HasConstantCamIntrinsics(camera_id)) {
                problem.SetParameterBlockConstant(camera.params.data());
            }
            else {
                std::vector<int> const_camera_params;

                if (!options.refine_focal_length) {
                    const span<const size_t> params_idxs = camera.FocalLengthIdxs();
                    const_camera_params.insert(
                        const_camera_params.end(), params_idxs.begin(), params_idxs.end());
                }
                if (!options.refine_principal_point) {
                    const span<const size_t> params_idxs = camera.PrincipalPointIdxs();
                    const_camera_params.insert(
                        const_camera_params.end(), params_idxs.begin(), params_idxs.end());
                }
                if (!options.refine_extra_params) {
                    const span<const size_t> params_idxs = camera.ExtraParamsIdxs();
                    const_camera_params.insert(
                        const_camera_params.end(), params_idxs.begin(), params_idxs.end());
                }

                if (const_camera_params.size() > 0) {
                    SetSubsetManifold(static_cast<int>(camera.params.size()),
                        const_camera_params,
                        &problem,
                        camera.params.data());
                }
            }
        }
    }

void ParameterizePoints(
    const BundleAdjustmentConfig& config,
    std::vector<struct Camera>& cameraList,
    std::vector<class Image>& imageList,
    std::unordered_map<point3D_t, Eigen::Vector3d>& objPts,
    ceres::Problem& problem) 
{
    std::unordered_map<point3D_t, int>objTrack;
    for (const auto&img: imageList)
    {
        for (const auto&[ptId,pt2d]:img.featPts)
        {
            if (objTrack.count(ptId)==0)
            {
                objTrack[ptId] = 1;
            }
            else
            {
                objTrack[ptId] += 1;
            }
        }
    }
    for (const auto& [point3D_id, num_observations] : objTrack)
    {
        Eigen::Vector3d& point3D = objPts[point3D_id];
        //if (num_observations > some_config) // freezen the point3d you want
        {
            problem.SetParameterBlockConstant(point3D.data());
        }
    }
    for (const point3D_t point3D_id : config.ConstantPoints()) 
    {
        Eigen::Vector3d& point3D = objPts[point3D_id];
        problem.SetParameterBlockConstant(point3D.data());
    }
}

    class DefaultBundleAdjuster : public BundleAdjuster {
    public:
        DefaultBundleAdjuster(BundleAdjustmentOptions options,
            BundleAdjustmentConfig config,
            std::vector<struct Camera>& cameraList,
            std::vector<class Image>& imageList,
            std::unordered_map<point3D_t, Eigen::Vector3d>& objPts)
            : BundleAdjuster(std::move(options), std::move(config)),
            loss_function_(std::unique_ptr<ceres::LossFunction>(
                options_.CreateLossFunction())) {
            ceres::Problem::Options problem_options;
            problem_options.loss_function_ownership = ceres::DO_NOT_TAKE_OWNERSHIP;
            problem_ = std::make_shared<ceres::Problem>(problem_options);

            // Set up problem
            // Warning: AddPointsToProblem assumes that AddImageToProblem is called
            // first. Do not change order of instructions!
            for (const image_t image_id : config_.Images()) {
                AddImageToProblem(image_id, cameraList, imageList, objPts);
            }
            //for (const auto point3D_id : config_.VariablePoints()) {
            //    AddPointToProblem(point3D_id, cameraList, imageList, objPts);
            //}
            //for (const auto point3D_id : config_.ConstantPoints()) {
            //    AddPointToProblem(point3D_id, cameraList, imageList, objPts);
            //}

            ParameterizeCameras(options_, config_, cameraList, imageList, objPts, *problem_);
            ParameterizePoints(config_, cameraList, imageList, objPts, *problem_);
        }

        ceres::Solver::Summary Solve() override {
            ceres::Solver::Summary summary;
            if (problem_->NumResiduals() == 0) {
                return summary;
            }

            const ceres::Solver::Options solver_options =
                options_.CreateSolverOptions(config_, *problem_);

            ceres::Solve(solver_options, problem_.get(), &summary);

            if (options_.print_summary || VLOG_IS_ON(1)) {
                PrintSolverSummary(summary, "Bundle adjustment report");
            }

            return summary;
        }

        std::shared_ptr<ceres::Problem>& Problem() override { return problem_; }

        void AddImageToProblem(const image_t image_id,
            std::vector<struct Camera>& cameraList,
            std::vector<class Image>& imageList,
            std::unordered_map<point3D_t, Eigen::Vector3d>& objPts) {
            class Image& image = imageList[image_id];
            struct Camera& camera = cameraList[image.CameraId()];

            // CostFunction assumes unit quaternions.
            image.CamFromWorld().rotation.normalize();

            double* cam_from_world_rotation =
                image.CamFromWorld().rotation.coeffs().data();
            double* cam_from_world_translation =
                image.CamFromWorld().translation.data();
            double* camera_params = camera.params.data();

            const bool constant_cam_pose = !options_.refine_extrinsics || config_.HasConstantCamPose(image_id);

            for (const auto& [ptId, point2D] : image.featPts)
            {
                auto& point3D = objPts[ptId];

                if (constant_cam_pose) {
                    problem_->AddResidualBlock(
                        CreateCameraCostFunction<ReprojErrorConstantPoseCostFunctor>(         //const T* const point3D, const T* const camera_params,
                            camera.model_id, point2D, image.CamFromWorld()),
                        loss_function_.get(),
                        point3D.data(),
                        camera_params);
                }
                else {
                    problem_->AddResidualBlock(
                        CreateCameraCostFunction<ReprojErrorCostFunctor>(camera.model_id,
                            point2D),
                        loss_function_.get(),
                        cam_from_world_rotation,
                        cam_from_world_translation,
                        point3D.data(),
                        camera_params);
                }
            }

            // Set pose parameterization.
            if (!constant_cam_pose) 
            {
                SetQuaternionManifold(problem_.get(), cam_from_world_rotation);
                if (config_.HasConstantCamPositions(image_id)) {
                    const std::vector<int>& constant_position_idxs =
                        config_.ConstantCamPositions(image_id);
                    SetSubsetManifold(3,
                        constant_position_idxs,
                        problem_.get(),
                        cam_from_world_translation);
                }
            }
        }

        void AddPointToProblem(const point3D_t point3D_id,
            std::vector<struct Camera>& cameraList,
            std::vector<class Image>& imageList,
            std::unordered_map<point3D_t, Eigen::Vector3d>& objPts) {
            auto& point3D = objPts[point3D_id];

            for (auto& image : imageList) 
            {
                if (image.featPts.count(point3D_id) > 0)
                {
                    Camera& camera = cameraList[image.CameraId()];
                    const auto& point2D = image.featPts[point3D_id];
                    // CostFunction assumes unit quaternions.
                    image.CamFromWorld().rotation.normalize();
                    //// We do not want to refine the camera of images that are not
                    //// part of `constant_image_ids_`, `constant_image_ids_`,
                    //// `constant_x_image_ids_`.
                    //if (camera_ids_.count(image.CameraId()) == 0) {
                    //    camera_ids_.insert(image.CameraId());
                    //    config_.SetConstantCamIntrinsics(image.CameraId());
                    //}
                    problem_->AddResidualBlock(
                        CreateCameraCostFunction<ReprojErrorConstantPoseCostFunctor>(
                            camera.model_id, point2D, image.CamFromWorld()),
                        loss_function_.get(),
                        point3D.data(),
                        camera.params.data());
                }
            }
        }

    private:
        std::shared_ptr<ceres::Problem> problem_;
        std::unique_ptr<ceres::LossFunction> loss_function_;
    };

    class RigBundleAdjuster : public BundleAdjuster {
    public:
        RigBundleAdjuster(BundleAdjustmentOptions options,
            RigBundleAdjustmentOptions rig_options,
            BundleAdjustmentConfig config,
            std::vector<struct Camera>& cameraList,
            std::vector<class Image>& imageList,
            std::unordered_map<point3D_t, Eigen::Vector3d>& objPts,
            std::vector<CameraRig>& camera_rigs)
            : BundleAdjuster(std::move(options), std::move(config)),
            rig_options_(rig_options),
            cameraList_(cameraList),
            imageList_(imageList),
            objPts_(objPts),
            loss_function_(std::unique_ptr<ceres::LossFunction>(
                options_.CreateLossFunction())) {
            // Check the validity of the provided camera rigs.
            std::unordered_set<camera_t> rig_camera_ids;
            for (CameraRig& camera_rig : camera_rigs) {
                camera_rig.Check(cameraList_, imageList_);
                for (const auto& camera_id : camera_rig.GetCameraIds()) {
                    if (rig_camera_ids.count(camera_id)!= 0)
                    {
                        LOG_ERR_OUT << "Camera must not be part of multiple camera rigs";
                    }
                    rig_camera_ids.insert(camera_id);
                }

                for (const auto& snapshot : camera_rig.Snapshots()) {
                    for (const auto& image_id : snapshot) {
                        if (image_id_to_camera_rig_.count(image_id) != 0)
                        {
                            LOG_ERR_OUT << "Image must not be part of multiple camera rigs";
                        }
                        image_id_to_camera_rig_.emplace(image_id, &camera_rig);
                    }
                }
            }

            ceres::Problem::Options problem_options;
            problem_options.loss_function_ownership = ceres::DO_NOT_TAKE_OWNERSHIP;
            problem_ = std::make_shared<ceres::Problem>(problem_options);

            ExtractRigsFromWorld(cameraList_,imageList_, camera_rigs);

            for (const image_t image_id : config_.Images()) {
                AddImageToProblem(image_id, cameraList_,imageList_, objPts_);
            }
            for (const auto point3D_id : config_.VariablePoints()) {
                AddPointToProblem(point3D_id, cameraList_, imageList_, objPts_);
            }
            for (const auto point3D_id : config_.ConstantPoints()) {
                AddPointToProblem(point3D_id, cameraList_, imageList_, objPts_);
            }

            ParameterizeCameras(
                options_, config_, cameraList_, imageList_, objPts_,*problem_);
            ParameterizeCameraRigs();
            ParameterizePoints(
                config_, cameraList_, imageList_, objPts_, *problem_);
        }

        ceres::Solver::Summary Solve() override {
            ceres::Solver::Summary summary;
            if (problem_->NumResiduals() == 0) {
                return summary;
            }

            const ceres::Solver::Options solver_options =
                options_.CreateSolverOptions(config_, *problem_);

            ceres::Solve(solver_options, problem_.get(), &summary);

            if (options_.print_summary || VLOG_IS_ON(1)) {
                PrintSolverSummary(summary, "Rig Bundle adjustment report");
            }

            ComputeCamsFromWorld();

            return summary;
        }

        std::shared_ptr<ceres::Problem>& Problem() override { return problem_; }

        void AddImageToProblem(const image_t image_id,
            std::vector<struct Camera>& cameraList,
            std::vector<class Image>& imageList,
            std::unordered_map<point3D_t, Eigen::Vector3d>& objPts) {
            const double max_squared_reproj_error =
                rig_options_.max_reproj_error * rig_options_.max_reproj_error;

            Image& image = imageList[image_id];
            Camera& camera = *image.CameraPtr();

            const bool constant_cam_pose = config_.HasConstantCamPose(image_id);
            const bool constant_cam_position =
                config_.HasConstantCamPositions(image_id);

            double* camera_params = camera.params.data();
            double* cam_from_rig_rotation = nullptr;
            double* cam_from_rig_translation = nullptr;
            double* rig_from_world_rotation = nullptr;
            double* rig_from_world_translation = nullptr;
            CameraRig* camera_rig = nullptr;
            Eigen::Matrix3x4d cam_from_world_mat = Eigen::Matrix3x4d::Zero();

            if (image_id_to_camera_rig_.count(image_id) > 0) {
                if (constant_cam_pose)
                {
                    LOG_ERR_OUT<< "Images contained in a camera rig must not have constant pose";
                }
                if (constant_cam_position)
                {
                    LOG_ERR_OUT << "Images contained in a camera rig must not have constant tvec";
                }
                camera_rig = image_id_to_camera_rig_.at(image_id);
                Rigid3d& rig_from_world = *image_id_to_rig_from_world_.at(image_id);
                rig_from_world_rotation = rig_from_world.rotation.coeffs().data();
                rig_from_world_translation = rig_from_world.translation.data();
                Rigid3d& cam_from_rig = camera_rig->CamFromRig(image.CameraId());
                cam_from_rig_rotation = cam_from_rig.rotation.coeffs().data();
                cam_from_rig_translation = cam_from_rig.translation.data();
                cam_from_world_mat = (cam_from_rig * rig_from_world).ToMatrix();
            }
            else {
                // CostFunction assumes unit quaternions.
                image.CamFromWorld().rotation.normalize();
                cam_from_rig_rotation = image.CamFromWorld().rotation.coeffs().data();
                cam_from_rig_translation = image.CamFromWorld().translation.data();
            }

            // Collect cameras for final parameterization.
            if (!image.HasCameraId())
            {
                LOG_ERR_OUT;
            }
            camera_ids_.insert(image.CameraId());

            // Add residuals to bundle adjustment problem.
            for (auto& [ptId, point2D] : image.featPts) {

                auto& point3D = objPts[ptId];
                if (camera_rig != nullptr &&
                    CalculateSquaredReprojectionError(
                        point2D, point3D, cam_from_world_mat, camera) >
                    max_squared_reproj_error) {
                    continue;
                }

     
                if (camera_rig == nullptr) {
                    if (constant_cam_pose) {
                        problem_->AddResidualBlock(
                            CreateCameraCostFunction<ReprojErrorConstantPoseCostFunctor>(
                                camera.model_id, point2D, image.CamFromWorld()),
                            loss_function_.get(),
                            point3D.data(),
                            camera_params);
                    }
                    else {
                        problem_->AddResidualBlock(
                            CreateCameraCostFunction<ReprojErrorCostFunctor>(camera.model_id,
                                point2D),
                            loss_function_.get(),
                            cam_from_rig_rotation,     // rig == world
                            cam_from_rig_translation,  // rig == world
                            point3D.data(),
                            camera_params);
                    }
                }
                else {
                    problem_->AddResidualBlock(
                        CreateCameraCostFunction<RigReprojErrorCostFunctor>(camera.model_id,
                            point2D),
                        loss_function_.get(),
                        cam_from_rig_rotation,
                        cam_from_rig_translation,
                        rig_from_world_rotation,
                        rig_from_world_translation,
                        point3D.data(),
                        camera_params);
                }
            }

            {
                parameterized_cams_from_rig_rotations_.insert(cam_from_rig_rotation);

                if (camera_rig != nullptr) {
                    parameterized_cams_from_rig_rotations_.insert(rig_from_world_rotation);

                    // Set the relative pose of the camera constant if relative pose
                    // refinement is disabled or if it is the reference camera to avoid
                    // over- parameterization of the camera pose.
                    if (!rig_options_.refine_relative_poses ||
                        image.CameraId() == camera_rig->RefCameraId()) {
                        problem_->SetParameterBlockConstant(cam_from_rig_rotation);
                        problem_->SetParameterBlockConstant(cam_from_rig_translation);
                    }
                }

                // Set pose parameterization.
                if (!constant_cam_pose && constant_cam_position) {
                    const std::vector<int>& constant_position_idxs =
                        config_.ConstantCamPositions(image_id);
                    SetSubsetManifold(3,
                        constant_position_idxs,
                        problem_.get(),
                        cam_from_rig_translation);
                }
            }
        }

        void AddPointToProblem(const point3D_t point3D_id,
            std::vector<struct Camera>& cameraList,
            std::vector<class Image>& imageList,
            std::unordered_map<point3D_t, Eigen::Vector3d>& objPts) {
            auto& point3D = objPts[point3D_id];

     

            //??? Skip observations that were already added in `AddImageToProblem`. ----
            for (size_t i = 0;i< imageList.size();i++)
            {
                Image& image = imageList[i];
                Camera& camera = cameraList[image.CameraId()];
                if (image.featPts.count(point3D_id) != 0)
                {
                    const auto& point2D = image.featPts[point3D_id];
                    problem_->AddResidualBlock(
                        CreateCameraCostFunction<ReprojErrorConstantPoseCostFunctor>(
                            camera.model_id, point2D, image.CamFromWorld()),
                        loss_function_.get(),
                        point3D.data(),
                        camera.params.data());
                }
            }
        }

        void ExtractRigsFromWorld(
            const std::vector<struct Camera>& cameraList,
            const std::vector<class Image>& imageList,
            const std::vector<CameraRig>& camera_rigs) {
            rigs_from_world_.reserve(camera_rigs.size());
            for (const auto& camera_rig : camera_rigs) {
                rigs_from_world_.emplace_back();
                auto& rig_from_world = rigs_from_world_.back();
                const size_t num_snapshots = camera_rig.NumSnapshots();
                rig_from_world.resize(num_snapshots);
                for (size_t snapshot_idx = 0; snapshot_idx < num_snapshots;
                    ++snapshot_idx) {
                    rig_from_world[snapshot_idx] =
                        camera_rig.ComputeRigFromWorld(snapshot_idx);
                    for (const auto image_id : camera_rig.Snapshots()[snapshot_idx]) {
                        image_id_to_rig_from_world_.emplace(image_id,
                            &rig_from_world[snapshot_idx]);
                    }
                }
            }
        }

        void ComputeCamsFromWorld() {
            
        }

        void ParameterizeCameraRigs() {
            for (double* cam_from_rig_rotation :
                parameterized_cams_from_rig_rotations_) {
                SetQuaternionManifold(problem_.get(), cam_from_rig_rotation);
            }
        }

    private:
        const RigBundleAdjustmentOptions rig_options_;


        std::vector<struct Camera>& cameraList_;
        std::vector<class Image>& imageList_;
        std::unordered_map<point3D_t, Eigen::Vector3d>& objPts_;
        std::shared_ptr<ceres::Problem> problem_;
        std::unique_ptr<ceres::LossFunction> loss_function_;

        std::unordered_set<camera_t> camera_ids_;
        std::unordered_map<point3D_t, size_t> point3D_num_observations_;

        // Mapping from images to camera rigs.
        std::unordered_map<image_t, CameraRig*> image_id_to_camera_rig_;
        std::unordered_map<image_t, Rigid3d*> image_id_to_rig_from_world_;

        // For each camera rig, the absolute camera rig poses for all snapshots.
        std::vector<std::vector<Rigid3d>> rigs_from_world_;

        // The Quaternions added to the problem, used to set the local
        // parameterization once after setting up the problem.
        std::unordered_set<double*> parameterized_cams_from_rig_rotations_;
    };

}  // namespace

std::unique_ptr<BundleAdjuster> CreateDefaultBundleAdjuster(
    BundleAdjustmentOptions options,
    BundleAdjustmentConfig config,
    std::vector<struct Camera>& cameraList,
    std::vector<class Image>& imageList,
    std::unordered_map<point3D_t, Eigen::Vector3d>& objPts) {
    return std::make_unique<DefaultBundleAdjuster>(
        std::move(options), std::move(config), cameraList, imageList, objPts);
}

std::unique_ptr<BundleAdjuster> CreateRigBundleAdjuster(
    BundleAdjustmentOptions options,
    RigBundleAdjustmentOptions rig_options,
    BundleAdjustmentConfig config,
    std::vector<struct Camera>& cameraList,
    std::vector<class Image>& imageList,
    std::unordered_map<point3D_t, Eigen::Vector3d>& objPts,
    std::vector<CameraRig>& camera_rigs) {
    return std::make_unique<RigBundleAdjuster>(std::move(options),
        rig_options,
        std::move(config),
        cameraList, imageList, objPts,
        camera_rigs);
}


void PrintSolverSummary(const ceres::Solver::Summary& summary,
    const std::string& header) {
    if (VLOG_IS_ON(3)) {
        LOG(INFO) << summary.FullReport();
    }

    std::ostringstream log;
    log << header << "\n";
    log << std::right << std::setw(16) << "Residuals : ";
    log << std::left << summary.num_residuals_reduced << "\n";

    log << std::right << std::setw(16) << "Parameters : ";
    log << std::left << summary.num_effective_parameters_reduced << "\n";

    log << std::right << std::setw(16) << "Iterations : ";
    log << std::left
        << summary.num_successful_steps + summary.num_unsuccessful_steps << "\n";

    log << std::right << std::setw(16) << "Time : ";
    log << std::left << summary.total_time_in_seconds << " [s]\n";

    log << std::right << std::setw(16) << "Initial cost : ";
    log << std::right << std::setprecision(6)
        << std::sqrt(summary.initial_cost / summary.num_residuals_reduced)
        << " [px]\n";

    log << std::right << std::setw(16) << "Final cost : ";
    log << std::right << std::setprecision(6)
        << std::sqrt(summary.final_cost / summary.num_residuals_reduced)
        << " [px]\n";

    log << std::right << std::setw(16) << "Termination : ";

    std::string termination = "";

    switch (summary.termination_type) {
    case ceres::CONVERGENCE:
        termination = "Convergence";
        break;
    case ceres::NO_CONVERGENCE:
        termination = "No convergence";
        break;
    case ceres::FAILURE:
        termination = "Failure";
        break;
    case ceres::USER_SUCCESS:
        termination = "User success";
        break;
    case ceres::USER_FAILURE:
        termination = "User failure";
        break;
    default:
        termination = "Unknown";
        break;
    }

    log << std::right << termination << "\n\n";
    LOG(INFO) << log.str();
}
