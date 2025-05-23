

#ifndef CERES_EXAMPLES_FCXCYK1_REPROJECTION_ERROR_H_
#define CERES_EXAMPLES_FCXCYK1_REPROJECTION_ERROR_H_
#include <vector>
#include <numeric>
#include "ceres/rotation.h"
#include "optiModel.h"
namespace ba {
    struct Fcxcyk1ReprojectionError {
        Fcxcyk1ReprojectionError(double observed_x, double observed_y)
            : observed_x(observed_x), observed_y(observed_y) {}

        template <typename T>
        bool operator()(const T* const camera,
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

            // Apply second and fourth order radial distortion.
            const T& l1 = camera[3];
            const T r2 = xp * xp + yp * yp;
            const T distortion = 1.0 + r2 * l1;

            // Compute final projected point position.
            const T& focal = camera[0];
            const T& cx = camera[1];
            const T& cy = camera[2];
            const T predicted_x = focal * distortion * xp + cx;
            const T predicted_y = focal * distortion * yp + cy;

            // The error is the difference between the predicted and observed position.
            residuals[0] = predicted_x - observed_x;
            residuals[1] = predicted_y - observed_y;

            return true;
        }

        // Factory to hide the construction of the CostFunction object from
        // the client code.
        static ceres::CostFunction* Create(const double observed_x,
            const double observed_y) {
            return (new ceres::AutoDiffCostFunction<Fcxcyk1ReprojectionError, 2, 4, 6, 3>(new Fcxcyk1ReprojectionError(observed_x, observed_y)));
        }

        double observed_x;
        double observed_y;

        static double figureErr(const double* objPts, const double* camerasParam, const double* imgRtParam, const double* imgPts, const int& imgPtsCnt, const int* point_index, const int* camera_index, const int* img_index)
        {
            std::vector<double>errs(imgPtsCnt);
            for (int i = 0; i < imgPtsCnt; i++)
            {
                const int& thisCameraIdx = camera_index[i];
                const int& thisObjIdx = point_index[i];
                const int& thisImgIdx = img_index[i];
                const double f = camerasParam[thisCameraIdx * 3];
                const double& cx = camerasParam[thisCameraIdx * 3 + 1];
                const double& cy = camerasParam[thisCameraIdx * 3 + 2];
                const double k1 = camerasParam[thisCameraIdx * 3 + 3];
                const double* rt = imgRtParam + 6 * thisImgIdx;
                const double* point = objPts + thisObjIdx * 3;

                double p[3];
                ceres::AngleAxisRotatePoint(rt, point, p);

                // camera[3,4,5] are the translation.
                p[0] += rt[3];
                p[1] += rt[4];
                p[2] += rt[5];

                // Compute the center of distortion. The sign change comes from
                // the camera model that Noah Snavely's Bundler assumes, whereby
                // the camera coordinate system has a negative z axis.
                const double xp = p[0] / p[2];
                const double yp = p[1] / p[2];

                const double r2 = xp * xp + yp * yp;
                const double distortion = 1.0 + r2 * k1;

                const double predicted_x = f * distortion * xp + cx;
                const double predicted_y = f * distortion * yp + cy;

                // The error is the difference between the predicted and observed position.
                double diffx = predicted_x - imgPts[2 * i];
                double diffy = predicted_y - imgPts[2 * i + 1];
                errs[i] = sqrt(diffx * diffx + diffy * diffy);
            }
            return std::accumulate(errs.begin(), errs.end(), 0.) / errs.size();
        }
    };

    struct Fcxcyk1ReprojectionErrorWithQuaternions {
        Fcxcyk1ReprojectionErrorWithQuaternions(double observed_x, double observed_y)
            : observed_x(observed_x), observed_y(observed_y) {}

        template <typename T>
        bool operator()(const T* const camera,
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

            // Apply second and fourth order radial distortion.
            const T& k1 = camera[3];

            const T r2 = xp * xp + yp * yp;
            const T distortion = 1.0 + r2 * k1;

            // Compute final projected point position.
            const T& focal = camera[0];
            const T& cx = camera[1];
            const T& cy = camera[2];
            const T predicted_x = focal * distortion * xp + cx;
            const T predicted_y = focal * distortion * yp + cy;

            // The error is the difference between the predicted and observed position.
            residuals[0] = predicted_x - observed_x;
            residuals[1] = predicted_y - observed_y;

            return true;
        }

        static double figureErr(const double* objPts, const double* camerasParam, const double* imgRtParam, const double* imgPts, const int& imgPtsCnt, const int* point_index, const int* camera_index, const int* img_index)
        {
            std::vector<double>errs(imgPtsCnt);
            for (int i = 0; i < imgPtsCnt; i++)
            {
                const int& thisCameraIdx = camera_index[i];
                const int& thisObjIdx = point_index[i];
                const int& thisImgIdx = img_index[i];
                const double f = camerasParam[thisCameraIdx * 3];
                const double& cx = camerasParam[thisCameraIdx * 3 + 1];
                const double& cy = camerasParam[thisCameraIdx * 3 + 2];
                const double k1 = camerasParam[thisCameraIdx * 3 + 1];
                const double* rt = imgRtParam + 7 * thisImgIdx;
                const double* point = objPts + thisObjIdx * 3;

                double p[3];
                ceres::QuaternionRotatePoint(rt, point, p);

                // camera[3,4,5] are the translation.
                p[0] += rt[4];
                p[1] += rt[5];
                p[2] += rt[6];

                // Compute the center of distortion. The sign change comes from
                // the camera model that Noah Snavely's Bundler assumes, whereby
                // the camera coordinate system has a negative z axis.
                const double xp = p[0] / p[2];
                const double yp = p[1] / p[2];

                const double r2 = xp * xp + yp * yp;
                const double distortion = 1.0 + r2 * k1;

                const double predicted_x = f * distortion * xp + cx;
                const double predicted_y = f * distortion * yp + cy;

                // The error is the difference between the predicted and observed position.
                double diffx = predicted_x - imgPts[2 * i];
                double diffy = predicted_y - imgPts[2 * i + 1];
                errs[i] = sqrt(diffx * diffx + diffy * diffy);
            }
            return std::accumulate(errs.begin(), errs.end(), 0.) / errs.size();
        }
        static ceres::CostFunction* Create(const double observed_x,
            const double observed_y) {
            return (
                new ceres::AutoDiffCostFunction<Fcxcyk1ReprojectionErrorWithQuaternions,
                4,
                2,
                7,
                3>(
                    new Fcxcyk1ReprojectionErrorWithQuaternions(observed_x,
                        observed_y)));
        }

        double observed_x;
        double observed_y;
    };

}  // namespace examples
  // namespace ceres

#endif  // CERES_EXAMPLES_SNAVELY_REPROJECTION_ERROR_H_
