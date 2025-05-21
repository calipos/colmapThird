#include <filesystem>
#include <iostream>
#include <vector>
#include "BALog.h"
#include "cameras.h"
#include "imagePt.h"
#include "objectPt.h"
#include "bal_problem.h"
#include "ceres/ceres.h"
#include "snavely_reprojection_error.h"
#include "optiModel.h"


bool readColmapResult(const std::filesystem::path& dataPath,
    std::map<int, std::array<double, 3>>& colmapObjPts,
    std::vector<col::ImgPt>& colmapImgPts,
    std::map<int, col::Camera>& cameras)
{
	const std::filesystem::path& camerasTXTpath = dataPath / "cameras.txt";
	const std::filesystem::path& imagesTXTpath = dataPath / "images.txt";
	const std::filesystem::path& points3DTXTpath = dataPath / "points3D.txt";
    if (!std::filesystem::exists(camerasTXTpath))
    {
        LOG_OUT << "camerasTXTpath not found: " << camerasTXTpath;
        return false;
    }
    if (!std::filesystem::exists(imagesTXTpath))
    {
        LOG_OUT << "imagesTXTpath not found: " << imagesTXTpath;
        return false;
    }
    if (!std::filesystem::exists(points3DTXTpath))
    {
        LOG_OUT << "points3DTXTpath not found: " << points3DTXTpath;
        return false;
    }
	colmapObjPts = readPoints3DFromTXT(points3DTXTpath);
	colmapImgPts = readImgPtsFromTXT(imagesTXTpath);
	cameras = readCamerasFromTXT(camerasTXTpath);
	return true;
}

bool use_quaternions = true;
bool use_HuberLoss = true;
bool use_manifolds = false;
std::string blocks_for_inner_iterations = "";
std::string ordering = "automatic";
auto cameraOptimModel = ba::OptiModel::fcxcyk1;
namespace ba 
{
    void SetLinearSolver(ceres::Solver::Options* options) {
        options->linear_solver_type;
        options->preconditioner_type;
        options->visibility_clustering_type;
        options->sparse_linear_algebra_library_type;
        options->dense_linear_algebra_library_type;
        options->linear_solver_ordering_type;
        options->use_explicit_schur_complement;
        options->use_mixed_precision_solves;
        options->max_num_refinement_iterations;
        options->max_linear_solver_iterations;
        options->use_spse_initialization;
        options->spse_tolerance;
        options->max_num_spse_iterations;
    }

    void SetOrdering(BALProblem* bal_problem, ceres::Solver::Options* options) {
        const int num_points = bal_problem->num_points();
        const int point_block_size = bal_problem->point_block_size();
        double* points = bal_problem->mutable_points();

        const int num_cameras = bal_problem->num_cameras();
        const int camera_block_size = bal_problem->camera_block_size();
        double* cameras = bal_problem->mutable_cameras();

        if (options->use_inner_iterations) 
        {
            if (blocks_for_inner_iterations .compare("cameras")==0) {
                LOG(INFO) << "Camera blocks for inner iterations";
                    options->inner_iteration_ordering =
                    std::make_shared<ceres::ParameterBlockOrdering>();
                    for (int i = 0; i < num_cameras; ++i) {
                        options->inner_iteration_ordering->AddElementToGroup(
                            cameras + camera_block_size * i, 0);
                    }
            }
            else if (blocks_for_inner_iterations .compare("points")==0) {
                LOG(INFO) << "Point blocks for inner iterations";
                    options->inner_iteration_ordering =
                    std::make_shared<ceres::ParameterBlockOrdering>();
                    for (int i = 0; i < num_points; ++i) {
                        options->inner_iteration_ordering->AddElementToGroup(
                            points + point_block_size * i, 0);
                    }
            }
            else if (blocks_for_inner_iterations.compare("cameras,points")==0) {
                LOG(INFO) << "Camera followed by point blocks for inner iterations";
                    options->inner_iteration_ordering =
                    std::make_shared<ceres::ParameterBlockOrdering>();
                    for (int i = 0; i < num_cameras; ++i) {
                        options->inner_iteration_ordering->AddElementToGroup(
                            cameras + camera_block_size * i, 0);
                    }
                for (int i = 0; i < num_points; ++i) {
                    options->inner_iteration_ordering->AddElementToGroup(
                        points + point_block_size * i, 1);
                }
            }
            else if (blocks_for_inner_iterations.compare(
                "points,cameras")==0) {
                LOG(INFO) << "Point followed by camera blocks for inner iterations";
                    options->inner_iteration_ordering =
                    std::make_shared<ceres::ParameterBlockOrdering>();
                    for (int i = 0; i < num_cameras; ++i) {
                        options->inner_iteration_ordering->AddElementToGroup(
                            cameras + camera_block_size * i, 1);
                    }
                for (int i = 0; i < num_points; ++i) {
                    options->inner_iteration_ordering->AddElementToGroup(
                        points + point_block_size * i, 0);
                }
            }
            else if (blocks_for_inner_iterations.compare("automatic") == 0) {
                LOG(INFO) << "Choosing automatic blocks for inner iterations";
            }
            else {
                LOG(FATAL) << "Unknown block type for inner iterations: "
                    << blocks_for_inner_iterations;
            }
        }

        if (ordering.compare("automatic") == 0) {
            return;
        }

        auto* ordering = new ceres::ParameterBlockOrdering;

        // The points come before the cameras.
        for (int i = 0; i < num_points; ++i) {
            ordering->AddElementToGroup(points + point_block_size * i, 0);
        }

        for (int i = 0; i < num_cameras; ++i) {
            // When using axis-angle, there is a single parameter block for
            // the entire camera.
            ordering->AddElementToGroup(cameras + camera_block_size * i, 1);
        }

        options->linear_solver_ordering.reset(ordering);
    }

    void SetMinimizerOptions(ceres::Solver::Options* options) {
        options->max_num_iterations = 100;
        options->minimizer_progress_to_stdout = true;
        options->num_threads = 4;
        options->eta;
        options->max_solver_time_in_seconds;
        options->use_nonmonotonic_steps;
        options->minimizer_type = ceres::LINE_SEARCH;
        options->trust_region_strategy_type;
        options->use_inner_iterations;
    }

    void SetSolverOptionsFromFlags(BALProblem* bal_problem,
        ceres::Solver::Options* options) {
        SetMinimizerOptions(options);
        SetLinearSolver(options);
        SetOrdering(bal_problem, options);
    }


    void BuildProblem(BALProblem * bal_problem, ceres::Problem * problem)
    {
        const int point_block_size = bal_problem->point_block_size();
        const int camera_block_size = bal_problem->camera_block_size();
        double* points = bal_problem->mutable_points();
        double* cameras = bal_problem->mutable_cameras();

        const double* observations = bal_problem->observations();
        for (int i = 0; i < bal_problem->num_observations(); ++i) {
            ceres::CostFunction* cost_function;
            cost_function = (use_quaternions)
                ? SnavelyReprojectionErrorWithQuaternions::Create(
                    observations[2 * i + 0], observations[2 * i + 1])
                : SnavelyReprojectionError::Create(
                    observations[2 * i + 0], observations[2 * i + 1]);

            ceres::LossFunction* loss_function = use_HuberLoss ? new ceres::HuberLoss(1.0) : nullptr;

            // Each observation correponds to a pair of a camera and a point
            // which are identified by camera_index()[i] and point_index()[i]
            // respectively.
            double* camera =
                cameras + camera_block_size * bal_problem->camera_index()[i];
            double* point = points + point_block_size * bal_problem->point_index()[i];
            problem->AddResidualBlock(cost_function, loss_function, camera, point);
        }

        if (use_quaternions && use_manifolds) {
            ceres::Manifold* camera_manifold =
                new ceres::ProductManifold<ceres::QuaternionManifold, ceres::EuclideanManifold<6>>{};
            for (int i = 0; i < bal_problem->num_cameras(); ++i) {
                problem->SetManifold(cameras + camera_block_size * i, camera_manifold);
            }
        }
    }

        void SolveProblem(const ba::OptiModel& optiModel,
            const std::map<int, std::array<double, 3>>&colmapObjPts,
            const std::vector<col::ImgPt>&colmapImgPts,
            const std::map<int, col::Camera>&cameras) {
            ba::BALProblem bal_problem(optiModel,colmapObjPts, colmapImgPts, cameras, use_quaternions);
            ceres::Problem problem;

            srand(0);
            bal_problem.Perturb(0, 0, 0);

            ba::BuildProblem(&bal_problem, &problem);
            ceres::Solver::Options options;
            ba::SetSolverOptionsFromFlags(&bal_problem, &options);
            options.gradient_tolerance = 1e-16;
            options.function_tolerance = 1e-16;
            ceres::Solver::Summary summary;
            Solve(options, &problem, &summary);
            std::cout << summary.FullReport() << "\n";
        }
    }
int main()
{
    std::map<int, std::array<double, 3>> colmapObjPts;
    std::vector<col::ImgPt> colmapImgPts;
    std::map<int, col::Camera>cameras;
    readColmapResult("D:/repo/colmapThird/data", colmapObjPts, colmapImgPts, cameras);
    ba::SolveProblem(cameraOptimModel, colmapObjPts, colmapImgPts, cameras);
	return 0;
}