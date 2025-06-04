

#ifndef CERES_EXAMPLES_SNAVELY_REPROJECTION_ERROR_H_
#define CERES_EXAMPLES_SNAVELY_REPROJECTION_ERROR_H_
#include <vector>
#include <numeric>
#include "ceres/rotation.h"
#include "optiModel.h"
namespace ba {


// Templated pinhole camera model for used with Ceres.  The camera is
// parameterized using 9 parameters: 3 for rotation, 3 for translation, 1 for
// focal length and 2 for radial distortion. The principal point is not modeled
// (i.e. it is assumed be located at the image center).
struct SnavelyReprojectionError {
  SnavelyReprojectionError(double observed_x, double observed_y)
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
    const T& l1 = camera[1];
    const T& l2 = camera[2];
    const T r2 = xp * xp + yp * yp;
    const T distortion = 1.0 + r2 * (l1 + l2 * r2);

    // Compute final projected point position.
    const T& focal = camera[0];
    const T predicted_x = focal * distortion * xp;
    const T predicted_y = focal * distortion * yp;

    // The error is the difference between the predicted and observed position.
    residuals[0] = predicted_x - observed_x;
    residuals[1] = predicted_y - observed_y;

    return true;
  }

  // Factory to hide the construction of the CostFunction object from
  // the client code.
  static ceres::CostFunction* Create(const double observed_x,
                                     const double observed_y) {
      return (new ceres::AutoDiffCostFunction<SnavelyReprojectionError, 2, 3, 6, 3>(new SnavelyReprojectionError(observed_x, observed_y)));
  }

  double observed_x;
  double observed_y;

  static double figureErr(const double*objPts, const double* camerasParam, const double* imgRtParam, const double*imgPts,const int& imgPtsCnt, const int* point_index, const int* camera_index, const int* img_index)
  {
      std::vector<double>errs(imgPtsCnt);
      for (int i = 0; i < imgPtsCnt; i++)
      {
          const int& thisCameraIdx = camera_index[i];
          const int& thisObjIdx = point_index[i];
          const int& thisImgIdx = img_index[i];
          const double f = camerasParam[thisCameraIdx * 3];
          const double k1 = camerasParam[thisCameraIdx * 3 + 1];
          const double k2 = camerasParam[thisCameraIdx * 3 + 2];
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
          const double distortion = 1.0 + r2 * (k1 + k2 * r2);

          const double predicted_x = f * distortion * xp;
          const double predicted_y = f * distortion * yp;

          // The error is the difference between the predicted and observed position.
          double diffx = predicted_x - imgPts[2 * i];
          double diffy = predicted_y - imgPts[2 * i + 1];
          errs[i] = sqrt(diffx * diffx + diffy * diffy);
      }
      return std::accumulate(errs.begin(), errs.end(), 0.) / errs.size();
  }
};

// Templated pinhole camera model for used with Ceres.  The camera is
// parameterized using 10 parameters. 4 for rotation, 3 for
// translation, 1 for focal length and 2 for radial distortion. The
// principal point is not modeled (i.e. it is assumed be located at
// the image center).
struct SnavelyReprojectionErrorWithQuaternions {
  // (u, v): the position of the observation with respect to the image
  // center point.
  SnavelyReprojectionErrorWithQuaternions(double observed_x, double observed_y)
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
    const T& l1 = camera[1];
    const T& l2 = camera[2];

    const T r2 = xp * xp + yp * yp;
    const T distortion = 1.0 + r2 * (l1 + l2 * r2);

    // Compute final projected point position.
    const T& focal = camera[0];
    const T predicted_x = focal * distortion * xp;
    const T predicted_y = focal * distortion * yp;

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
          const double k1 = camerasParam[thisCameraIdx * 3 + 1];
          const double k2 = camerasParam[thisCameraIdx * 3 + 2];
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
          const double distortion = 1.0 + r2 * (k1 + k2 * r2);

          const double predicted_x = f * distortion * xp;
          const double predicted_y = f * distortion * yp;

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
        new ceres::AutoDiffCostFunction<SnavelyReprojectionErrorWithQuaternions,
                                        2,
                                        3,
                                        7,
                                        3>(
            new SnavelyReprojectionErrorWithQuaternions(observed_x,
                                                        observed_y)));
  }

  double observed_x;
  double observed_y;
};

}  // namespace examples
  // namespace ceres

#endif  // CERES_EXAMPLES_SNAVELY_REPROJECTION_ERROR_H_
