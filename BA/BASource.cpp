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
#include "fk1_reprojection_error.h"
#include "fcxcyk1_reprojection_error.h"
#include "fcxcy_reprojection_error.h"
#include "fixcamera_reprojection_error.h"
#include "k1_reprojection_error.h"
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

auto cameraOptimModel = ba::OptiModel::fk1;
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
        options->max_num_iterations = 5000;
        options->minimizer_progress_to_stdout = true;
        options->num_threads = 1;
        options->eta;
        options->max_solver_time_in_seconds;
        options->use_nonmonotonic_steps;
        options->minimizer_type;// = ceres::LINE_SEARCH;
        options->trust_region_strategy_type;
        options->use_inner_iterations;
    }

    void SetSolverOptionsFromFlags(BALProblem* bal_problem, ceres::Solver::Options* options) {
        SetMinimizerOptions(options);
        SetLinearSolver(options);
        //SetOrdering(bal_problem, options);
    }

    void BuildProblem(BALProblem * bal_problem, ceres::Problem * problem)
    {
        double* points = bal_problem->mutable_points();
        double* cameras = bal_problem->mutable_cameras();
        double* imgRts = bal_problem->mutable_imgRts();
        const double* observations = bal_problem->observations();
        bal_problem->errorBefore = 0;
        bal_problem->errorAfter = 0;
        switch (bal_problem->optiModel_)
        {
        case ba::OptiModel::fk1k2:
            bal_problem->errorBefore = (bal_problem->use_quaternions_)?
                SnavelyReprojectionErrorWithQuaternions::figureErr(points, cameras, imgRts, observations, bal_problem->num_observations(), bal_problem->point_index(), bal_problem->camera_index(), bal_problem->img_index())
                : SnavelyReprojectionError::figureErr(points, cameras, imgRts,observations, bal_problem->num_observations(), bal_problem->point_index(), bal_problem->camera_index(), bal_problem->img_index());
            break;
        case ba::OptiModel::fk1:
            bal_problem->errorBefore = (bal_problem->use_quaternions_) ?
                Fk1ReprojectionErrorWithQuaternions::figureErr(points, cameras, imgRts, observations, bal_problem->num_observations(), bal_problem->point_index(), bal_problem->camera_index(), bal_problem->img_index())
                : Fk1ReprojectionError::figureErr(points, cameras, imgRts, observations, bal_problem->num_observations(), bal_problem->point_index(), bal_problem->camera_index(), bal_problem->img_index());
            break;
        case ba::OptiModel::fcxcyk1:
            bal_problem->errorBefore = (bal_problem->use_quaternions_) ?
                Fcxcyk1ReprojectionErrorWithQuaternions::figureErr(points, cameras, imgRts, observations, bal_problem->num_observations(), bal_problem->point_index(), bal_problem->camera_index(), bal_problem->img_index())
                : Fcxcyk1ReprojectionError::figureErr(points, cameras, imgRts, observations, bal_problem->num_observations(), bal_problem->point_index(), bal_problem->camera_index(), bal_problem->img_index());
            break;
        case ba::OptiModel::fixcamera1:
            bal_problem->errorBefore = (bal_problem->use_quaternions_) ?
                Fixcamera1ReprojectionErrorWithQuaternions::figureErr(points, imgRts, observations, bal_problem->num_observations(), bal_problem->point_index(), bal_problem->img_index())
                : Fixcamera1ReprojectionError::figureErr(points, imgRts, observations, bal_problem->num_observations(), bal_problem->point_index(), bal_problem->img_index());
            break;
        case ba::OptiModel::k1:
                bal_problem->errorBefore = (bal_problem->use_quaternions_) ?
                    K1ReprojectionErrorWithQuaternions::figureErr(points, cameras, imgRts, observations, bal_problem->num_observations(), bal_problem->point_index(), bal_problem->camera_index(), bal_problem->img_index())
                    : K1ReprojectionError::figureErr(points, cameras, imgRts, observations, bal_problem->num_observations(), bal_problem->point_index(), bal_problem->camera_index(), bal_problem->img_index());
                break;
        case ba::OptiModel::fcxcy:
            bal_problem->errorBefore = (bal_problem->use_quaternions_) ?
                FcxcyReprojectionErrorWithQuaternions::figureErr(points, cameras, imgRts, observations, bal_problem->num_observations(), bal_problem->point_index(), bal_problem->camera_index(), bal_problem->img_index())
                : FcxcyReprojectionError::figureErr(points, cameras, imgRts, observations, bal_problem->num_observations(), bal_problem->point_index(), bal_problem->camera_index(), bal_problem->img_index());
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
                cost_function = (bal_problem->use_quaternions_)
                    ? SnavelyReprojectionErrorWithQuaternions::Create(
                        observations[2 * i + 0], observations[2 * i + 1])
                    : SnavelyReprojectionError::Create(
                        observations[2 * i + 0], observations[2 * i + 1]);
                break;
            case ba::OptiModel::fk1:  
                cost_function = (bal_problem->use_quaternions_)
                    ? Fk1ReprojectionErrorWithQuaternions::Create(
                        observations[2 * i + 0], observations[2 * i + 1])
                    : Fk1ReprojectionError::Create(
                        observations[2 * i + 0], observations[2 * i + 1]);
                break;
            case ba::OptiModel::fcxcyk1:
                cost_function = (bal_problem->use_quaternions_)
                    ? Fcxcyk1ReprojectionErrorWithQuaternions::Create(
                        observations[2 * i + 0], observations[2 * i + 1])
                    : Fcxcyk1ReprojectionError::Create(
                        observations[2 * i + 0], observations[2 * i + 1]);
                break;
            case ba::OptiModel::fixcamera1:
                cost_function = (bal_problem->use_quaternions_)
                    ? Fixcamera1ReprojectionErrorWithQuaternions::Create(
                        observations[2 * i + 0], observations[2 * i + 1])
                    : Fixcamera1ReprojectionError::Create(
                        observations[2 * i + 0], observations[2 * i + 1]);
                break;
            case ba::OptiModel::k1:
                cost_function = (bal_problem->use_quaternions_)
                    ? K1ReprojectionErrorWithQuaternions::Create(
                        observations[2 * i + 0], observations[2 * i + 1])
                    : K1ReprojectionError::Create(
                        observations[2 * i + 0], observations[2 * i + 1]);
                break;
            case ba::OptiModel::fcxcy:
                cost_function = (bal_problem->use_quaternions_)
                    ? FcxcyReprojectionErrorWithQuaternions::Create(
                        observations[2 * i + 0], observations[2 * i + 1])
                    : FcxcyReprojectionError::Create(
                        observations[2 * i + 0], observations[2 * i + 1]);
                break;
            default: 
                break;
            }
            ceres::LossFunction* loss_function = use_HuberLoss ? new ceres::HuberLoss(1.0) : nullptr;
            //LOG_OUT << i << ": imgpt=(" << observations[2 * i + 0] << ", " << observations[2 * i + 1] << "; camera_index=" << bal_problem->camera_index()[i] << "; objPt_index=" << bal_problem->point_index()[i];            
            double* camera = cameras + getEachCameraParamCnt(bal_problem->optiModel_) * bal_problem->camera_index()[i];
            double* imgRt = imgRts + (bal_problem->use_quaternions_ ? 7 : 6) * bal_problem->img_index()[i];
            double* point = points +  3*bal_problem->point_index()[i];
            if (bal_problem->optiModel_==ba::OptiModel::fixcamera1)
            {
                problem->AddResidualBlock(cost_function, loss_function, imgRt, point);
            }
            else
            {
                problem->AddResidualBlock(cost_function, loss_function, camera, imgRt, point);
            }
        } 
       
        return;
    }

    void SolveProblem(const std::filesystem::path&dataPath,
        const ba::OptiModel& optiModel,
        const std::map<int, std::array<double, 3>>&colmapObjPts,
        std::vector<col::ImgPt>&colmapImgPts,
        const std::map<int, col::Camera>&cameras) {
        ba::BALProblem bal_problem(optiModel,colmapObjPts, colmapImgPts, cameras, use_quaternions);
        ceres::Problem problem;

        bal_problem.printBrief();
        srand(0);
        //bal_problem.Perturb(10, 10, 10);

        ba::BuildProblem(&bal_problem, &problem);
        ceres::Solver::Options options;
        ba::SetSolverOptionsFromFlags(&bal_problem, &options);
        options.gradient_tolerance = 1e-16;
        options.function_tolerance = 1e-16;
        ceres::Solver::Summary summary;
        Solve(options, &problem, &summary);

        switch (bal_problem.optiModel_)
        {
        case ba::OptiModel::fk1k2:
            bal_problem.errorAfter = (bal_problem.use_quaternions_) ? 
                SnavelyReprojectionErrorWithQuaternions::figureErr(bal_problem.mutable_points(), bal_problem.mutable_cameras(), bal_problem.mutable_imgRts(), bal_problem.observations(), bal_problem.num_observations(), bal_problem.point_index(), bal_problem.camera_index(), bal_problem.img_index())
                : SnavelyReprojectionError::figureErr(bal_problem.mutable_points(), bal_problem.mutable_cameras(), bal_problem.mutable_imgRts(), bal_problem.observations(), bal_problem.num_observations(), bal_problem.point_index(), bal_problem.camera_index(), bal_problem.img_index());
            break;
        case ba::OptiModel::fk1:
            bal_problem.errorAfter = (bal_problem.use_quaternions_) ?
                Fk1ReprojectionErrorWithQuaternions::figureErr(bal_problem.mutable_points(), bal_problem.mutable_cameras(), bal_problem.mutable_imgRts(), bal_problem.observations(), bal_problem.num_observations(), bal_problem.point_index(), bal_problem.camera_index(), bal_problem.img_index())
                : Fk1ReprojectionError::figureErr(bal_problem.mutable_points(), bal_problem.mutable_cameras(), bal_problem.mutable_imgRts(), bal_problem.observations(), bal_problem.num_observations(), bal_problem.point_index(), bal_problem.camera_index(), bal_problem.img_index());
            break;
        case ba::OptiModel::fcxcyk1:
            bal_problem.errorAfter = (bal_problem.use_quaternions_) ?
                Fcxcyk1ReprojectionErrorWithQuaternions::figureErr(bal_problem.mutable_points(), bal_problem.mutable_cameras(), bal_problem.mutable_imgRts(), bal_problem.observations(), bal_problem.num_observations(), bal_problem.point_index(), bal_problem.camera_index(), bal_problem.img_index())
                : Fcxcyk1ReprojectionError::figureErr(bal_problem.mutable_points(), bal_problem.mutable_cameras(), bal_problem.mutable_imgRts(), bal_problem.observations(), bal_problem.num_observations(), bal_problem.point_index(), bal_problem.camera_index(), bal_problem.img_index());
            break;
        case ba::OptiModel::fixcamera1:
            bal_problem.errorAfter = (bal_problem.use_quaternions_) ?
                Fixcamera1ReprojectionErrorWithQuaternions::figureErr(bal_problem.mutable_points(), bal_problem.mutable_imgRts(), bal_problem.observations(), bal_problem.num_observations(), bal_problem.point_index(), bal_problem.img_index())
                : Fixcamera1ReprojectionError::figureErr(bal_problem.mutable_points(), bal_problem.mutable_imgRts(), bal_problem.observations(), bal_problem.num_observations(), bal_problem.point_index(), bal_problem.img_index());
            break;
        case ba::OptiModel::k1:
            bal_problem.errorAfter = (bal_problem.use_quaternions_) ?
                K1ReprojectionErrorWithQuaternions::figureErr(bal_problem.mutable_points(), bal_problem.mutable_cameras(), bal_problem.mutable_imgRts(), bal_problem.observations(), bal_problem.num_observations(), bal_problem.point_index(), bal_problem.camera_index(), bal_problem.img_index())
                : K1ReprojectionError::figureErr(bal_problem.mutable_points(), bal_problem.mutable_cameras(), bal_problem.mutable_imgRts(), bal_problem.observations(), bal_problem.num_observations(), bal_problem.point_index(), bal_problem.camera_index(), bal_problem.img_index());
            break;
        case ba::OptiModel::fcxcy:
            bal_problem.errorAfter = (bal_problem.use_quaternions_) ?
                FcxcyReprojectionErrorWithQuaternions::figureErr(bal_problem.mutable_points(), bal_problem.mutable_cameras(), bal_problem.mutable_imgRts(), bal_problem.observations(), bal_problem.num_observations(), bal_problem.point_index(), bal_problem.camera_index(), bal_problem.img_index())
                : FcxcyReprojectionError::figureErr(bal_problem.mutable_points(), bal_problem.mutable_cameras(), bal_problem.mutable_imgRts(), bal_problem.observations(), bal_problem.num_observations(), bal_problem.point_index(), bal_problem.camera_index(), bal_problem.img_index());
            break;
        default:
            break;
        }
        
        for (int i = 0; i < colmapImgPts.size(); ++i) 
        {
            if (bal_problem.use_quaternions_)
            {
                const double* quaternion_cursor = bal_problem.imgRts() + 7 * i;
                ceres::QuaternionToAngleAxis(quaternion_cursor, &colmapImgPts[i].r[0]);
                colmapImgPts[i].t[0] = quaternion_cursor[4];
                colmapImgPts[i].t[1] = quaternion_cursor[5];
                colmapImgPts[i].t[2] = quaternion_cursor[6];
            }
            else
            {
                const double* rt = bal_problem.imgRts() + 6 * i;
                colmapImgPts[i].r[0] = rt[0];
                colmapImgPts[i].r[1] = rt[1];
                colmapImgPts[i].r[2] = rt[2];
                colmapImgPts[i].t[0] = rt[3];
                colmapImgPts[i].t[1] = rt[4];
                colmapImgPts[i].t[2] = rt[5];
            }
        }
        
        std::cout << summary.BriefReport() << "\n";
        //std::cout << summary.FullReport() << "\n";
        LOG_OUT << bal_problem.errorBefore;
        LOG_OUT << bal_problem.errorAfter;
        bal_problem.printBrief();
        bal_problem.saveResultJson(dataPath, colmapObjPts, colmapImgPts, cameras);
    }
}
struct x
{
    double x;
};
struct MyStruct
{
    x x()
    {
        return x();
    }
};
int main(int argc, char** argv)
{
    MyStruct a;
    std::filesystem::path  dataPath = "../data";
    if (argc>1)
    {
        dataPath = std::filesystem::path(argv[1]);
    }


    cameraOptimModel = ba::OptiModel::fcxcy;
    use_quaternions = false;
    std::map<int, std::array<double, 3>> colmapObjPts;
    std::vector<col::ImgPt> colmapImgPts;
    std::map<int, col::Camera>cameras;
    if (!readColmapResult(dataPath, colmapObjPts, colmapImgPts, cameras))
    {
        return -1;
    };
    ba::SolveProblem(dataPath,cameraOptimModel, colmapObjPts, colmapImgPts, cameras);
	return 0;
}