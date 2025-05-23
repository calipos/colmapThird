

#ifndef CERES_EXAMPLES_FIXCAMERA_REPROJECTION_ERROR_H_
#define CERES_EXAMPLES_FIXCAMERA_REPROJECTION_ERROR_H_
#include <vector>
#include <numeric>
#include "ceres/rotation.h"
#include "optiModel.h"
namespace ba {


    // Templated pinhole camera model for used with Ceres.  The camera is
    // parameterized using 9 parameters: 3 for rotation, 3 for translation, 1 for
    // focal length and 2 for radial distortion. The principal point is not modeled
    // (i.e. it is assumed be located at the image center).
    struct Fixcamera1ReprojectionError {
        Fixcamera1ReprojectionError(double observed_x, double observed_y)
            : observed_x(observed_x), observed_y(observed_y) {}

        template <typename T>
        bool operator()(
            const T* const imgRt,
            const T* const point,
            T* residuals) const {
            T p[3];
            ceres::AngleAxisRotatePoint(imgRt, point, p);

            p[0] += imgRt[3];
            p[1] += imgRt[4];
            p[2] += imgRt[5];

            const T xp = p[0] / p[2];
            const T yp = p[1] / p[2];
             

            // The error is the difference between the predicted and observed position.
            residuals[0] = xp - observed_x;
            residuals[1] = yp - observed_y;

            return true;
        }

        // Factory to hide the construction of the CostFunction object from
        // the client code.
        static ceres::CostFunction* Create(const double observed_x,
            const double observed_y) {
            return (new ceres::AutoDiffCostFunction<Fixcamera1ReprojectionError, 2, 6, 3>(new Fixcamera1ReprojectionError(observed_x, observed_y)));
        }

        double observed_x;
        double observed_y;

        static double figureErr(const double* objPts, const double* imgRtParam, const double* imgPts, const int& imgPtsCnt, const int* point_index, const int* img_index)
        {
            std::vector<double>errs(imgPtsCnt);
            for (int i = 0; i < imgPtsCnt; i++)
            {
                const int& thisObjIdx = point_index[i];
                const int& thisImgIdx = img_index[i];
                const double* rt = imgRtParam + 6 * thisImgIdx;
                const double* point = objPts + thisObjIdx * 3;

                double p[3];
                ceres::AngleAxisRotatePoint(rt, point, p);

                p[0] += rt[3];
                p[1] += rt[4];
                p[2] += rt[5];

                const double xp = p[0] / p[2];
                const double yp = p[1] / p[2];
                  
                double diffx = xp - imgPts[2 * i];
                double diffy = yp - imgPts[2 * i + 1];
                errs[i] = sqrt(diffx * diffx + diffy * diffy);
            }
            return std::accumulate(errs.begin(), errs.end(), 0.) / errs.size();
        }
    };

    struct Fixcamera1ReprojectionErrorWithQuaternions {
        Fixcamera1ReprojectionErrorWithQuaternions(double observed_x, double observed_y)
            : observed_x(observed_x), observed_y(observed_y) {}

        template <typename T>
        bool operator()(
            const T* const imgRt,
            const T* const point,
            T* residuals) const {
            T p[3];
            ceres::QuaternionRotatePoint(imgRt, point, p);

            p[0] += imgRt[4];
            p[1] += imgRt[5];
            p[2] += imgRt[6];

            const T xp = p[0] / p[2];
            const T yp = p[1] / p[2];

            residuals[0] = xp - observed_x;
            residuals[1] = yp - observed_y;

            return true;
        }

        static double figureErr(const double* objPts, const double* imgRtParam, const double* imgPts, const int& imgPtsCnt, const int* point_index, const int* img_index)
        {
            std::vector<double>errs(imgPtsCnt);
            for (int i = 0; i < imgPtsCnt; i++)
            {
                const int& thisObjIdx = point_index[i];
                const int& thisImgIdx = img_index[i];
                const double* rt = imgRtParam + 7 * thisImgIdx;
                const double* point = objPts + thisObjIdx * 3;

                double p[3];
                ceres::QuaternionRotatePoint(rt, point, p);

                p[0] += rt[4];
                p[1] += rt[5];
                p[2] += rt[6];

                const double xp = p[0] / p[2];
                const double yp = p[1] / p[2];

                double diffx = xp - imgPts[2 * i];
                double diffy = yp - imgPts[2 * i + 1];
                errs[i] = sqrt(diffx * diffx + diffy * diffy);
            }
            return std::accumulate(errs.begin(), errs.end(), 0.) / errs.size();
        }
        static ceres::CostFunction* Create(const double observed_x,
            const double observed_y) {
            return (
                new ceres::AutoDiffCostFunction<Fixcamera1ReprojectionErrorWithQuaternions,
                2,
                7,
                3>(
                    new Fixcamera1ReprojectionErrorWithQuaternions(observed_x,
                        observed_y)));
        }

        double observed_x;
        double observed_y;
    };

}  // namespace examples
  // namespace ceres

#endif  // CERES_EXAMPLES_SNAVELY_REPROJECTION_ERROR_H_
