#ifndef CERES_EXAMPLES_BAL_PROBLEM_H_
#define CERES_EXAMPLES_BAL_PROBLEM_H_
#include <map>
#include <string>
#include "cameras.h"
#include "imagePt.h"
#include "optiModel.h"
namespace ba {

    class BALProblem {
    public:
        explicit BALProblem(
            const ba::OptiModel& optiModel,
            const std::map<int, std::array<double, 3>>& colmapObjPts,
            const std::vector<col::ImgPt>& colmapImgPts,
            const std::map<int, col::Camera>& cameras, 
            bool use_quaternions);
        ~BALProblem();

        void WriteToFile(const std::string& filename) const;
        void WriteToPLYFile(const std::string& filename) const;
       


        // Perturb the camera pose and the geometry with random normal
        // numbers with corresponding standard deviations.
        void Perturb(const double rotation_sigma,
            const double translation_sigma,
            const double point_sigma);

        // clang-format off
        int camera_block_size()      const { return use_quaternions_ ? 10 : 9; }
        int point_block_size()       const { return 3; }
        int num_cameras()            const { return num_cameras_; }
        int num_image()            const { return img_Cnt_; }
        int num_points()             const { return num_points_; }
        int num_observations()       const { return num_observations_; }
        int num_parameters()         const { return num_parameters_; }
        const int* point_index()     const { return point_index_; }
        const int* camera_index()    const { return camera_index_; }
        const double* observations() const { return observations_; }
        const double* parameters()   const { return parameters_; }
        const double* cameras()      const { return parameters_; }
        double* mutable_cameras() { return parameters_; }
        // clang-format on
        double* mutable_points() {
            int eachCameraParamCnt = getEachCameraParamCnt(optiModel_);
            return parameters_ + eachCameraParamCnt* num_cameras_ 
                + num_image()* (use_quaternions_ ? 7 : 6)
                + +3 * num_points();
        }

        ba::OptiModel  optiModel_;
    private:
        int num_cameras_;
        int img_Cnt_;
        int num_points_;
        int num_observations_;
        int num_parameters_;
        bool use_quaternions_;

        int* point_index_;
        int* camera_index_;
        double* observations_;
        // The parameter vector is laid out as follows
        // [camera_1, ..., camera_n, point_1, ..., point_m]
        double* parameters_;
    };

}  // namespace ceres::examples

#endif  // CERES_EXAMPLES_BAL_PROBLEM_H_
