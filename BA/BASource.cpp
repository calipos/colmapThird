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

bool use_quaternions = false;
bool use_HuberLoss = false;

auto cameraOptimModel = ba::OptiModel::fk1k2;
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


    void SetMinimizerOptions(ceres::Solver::Options* options) {
        options->max_num_iterations = 1000;
        options->minimizer_progress_to_stdout = true;
        options->num_threads = 4;
        options->eta;
        options->max_solver_time_in_seconds;
        options->use_nonmonotonic_steps;
        options->minimizer_type;
        options->trust_region_strategy_type;
        options->use_inner_iterations;
    }

    void SetSolverOptionsFromFlags(BALProblem* bal_problem,
        ceres::Solver::Options* options) {
        SetMinimizerOptions(options);
        SetLinearSolver(options);
        //SetOrdering(bal_problem, options);
    }


    void BuildProblem(BALProblem * bal_problem, ceres::Problem * problem)
    {
        double* points = bal_problem->mutable_points();
        double* cameras = bal_problem->mutable_cameras();
        const double* observations = bal_problem->observations();
        double errorBefore = 0;
        switch (bal_problem->optiModel_)
        {
        case ba::OptiModel::fk1k2:
            errorBefore = (use_quaternions)?0
                : SnavelyReprojectionError::figureErr(points, cameras, observations, bal_problem->num_observations(), bal_problem->point_index(), bal_problem->camera_index());
            break;
        case ba::OptiModel::fk1:
            break;
        case ba::OptiModel::fcxcyk1:
            break;
        default:
            break;
        }

        for (int i = 0; i < bal_problem->num_observations(); ++i) 
        {
            ceres::CostFunction* cost_function=nullptr;
            switch (bal_problem->optiModel_)
            {
            case ba::OptiModel::fk1k2:
                cost_function = (use_quaternions)
                    ? SnavelyReprojectionErrorWithQuaternions::Create(
                        observations[2 * i + 0], observations[2 * i + 1])
                    : SnavelyReprojectionError::Create(
                        observations[2 * i + 0], observations[2 * i + 1]);
                break;
            case ba::OptiModel::fk1:  
                break;
            case ba::OptiModel::fcxcyk1: 
                break;
            default: 
                break;
            }


            

            ceres::LossFunction* loss_function = use_HuberLoss ? new ceres::HuberLoss(1.0) : nullptr;


            //LOG_OUT << i << ": imgpt=(" << observations[2 * i + 0] << ", " << observations[2 * i + 1] << "; camera_index=" << bal_problem->camera_index()[i] << "; objPt_index=" << bal_problem->point_index()[i];
            
            double* camera = cameras + getEachCameraParamCnt(bal_problem->optiModel_) * bal_problem->camera_index()[i];
            double* point = points +  3*bal_problem->point_index()[i];
            problem->AddResidualBlock(cost_function, loss_function, camera, point);
        } 
    }

    void SolveProblem(const ba::OptiModel& optiModel,
        const std::map<int, std::array<double, 3>>&colmapObjPts,
        const std::vector<col::ImgPt>&colmapImgPts,
        const std::map<int, col::Camera>&cameras) {
        ba::BALProblem bal_problem(optiModel,colmapObjPts, colmapImgPts, cameras, use_quaternions);
        ceres::Problem problem;

        srand(0);
        //bal_problem.Perturb(10, 10, 10);

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
    readColmapResult("../data", colmapObjPts, colmapImgPts, cameras);
    ba::SolveProblem(cameraOptimModel, colmapObjPts, colmapImgPts, cameras);
	return 0;
}