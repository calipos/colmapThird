#ifndef _CAMERA_H_
#define _CAMERA_H_
#include <vector>
#include <Eigen/Geometry>
#include "log.h"
#include "misc.h"
#include "types.h"
#include <ceres/jet.h>


#define MAKE_ENUM_CLASS_OVERLOAD_STREAM  \
    X(kInvalid,                -1 ) \
    X(kSimplePinhole,          0  ) \
    X(kPinhole,                1  ) \
    X(kSimpleRadial,           2  ) \
    X(kRadial,                 3  ) \
    X(kOpenCV,                 4  ) 
    //X(kOpenCVFisheye,          5  ) \
    //X(kFullOpenCV,             6  ) \
    //X(kFOV,                    7  ) \
    //X(kSimpleRadialFisheye,    8  ) \
    //X(kRadialFisheye,          9  ) \
    //X(kThinPrismFisheye,       10 ) 




#ifndef CAMERA_MODEL_CASES
#define CAMERA_MODEL_CASES                          \
  CAMERA_MODEL_CASE(SimplePinholeCameraModel)       \
  CAMERA_MODEL_CASE(PinholeCameraModel)             \
  CAMERA_MODEL_CASE(SimpleRadialCameraModel)        \
  CAMERA_MODEL_CASE(RadialCameraModel)              \
  CAMERA_MODEL_CASE(OpenCVCameraModel)
#endif


#define CAMERA_MODEL_SWITCH_CASES         \
  CAMERA_MODEL_CASES                      \
  default:                                \
    CAMERA_MODEL_DOES_NOT_EXIST_EXCEPTION \
    break;

#define CAMERA_MODEL_DOES_NOT_EXIST_EXCEPTION \
  throw std::domain_error("Camera model does not exist");

enum class CameraModelId : int {
#define X(name, value) name = value,
    MAKE_ENUM_CLASS_OVERLOAD_STREAM
#undef X
};


 

template <typename CameraModel>
struct BaseCameraModel {
    template <typename T>
    static inline bool HasBogusParams(const std::vector<T>& params,
        size_t width,
        size_t height,
        T min_focal_length_ratio,
        T max_focal_length_ratio,
        T max_extra_param)
    {
        return HasBogusPrincipalPoint(params, width, height) ||
            HasBogusFocalLength(params,
                width,
                height,
                min_focal_length_ratio,
                max_focal_length_ratio) ||
            HasBogusExtraParams(params, max_extra_param);
    }

    template <typename T>
    static inline bool HasBogusFocalLength(const std::vector<T>& params,
        size_t width,
        size_t height,
        T min_focal_length_ratio,
        T max_focal_length_ratio) 
    {
        const T inv_max_size = 1.0 / std::max(width, height);
        for (const size_t idx : CameraModel::focal_length_idxs) {
            const T focal_length_ratio = params[idx] * inv_max_size;
            if (focal_length_ratio < min_focal_length_ratio ||
                focal_length_ratio > max_focal_length_ratio) {
                return true;
            }
        }
        return false;
    }

    template <typename T>
    static inline bool HasBogusPrincipalPoint(const std::vector<T>& params,
        size_t width,
        size_t height)
    {
        const T cx = params[CameraModel::principal_point_idxs[0]];
        const T cy = params[CameraModel::principal_point_idxs[1]];
        return cx < 0 || cx > width || cy < 0 || cy > height;
    }

    template <typename T>
    static inline bool HasBogusExtraParams(const std::vector<T>& params,
        T max_extra_param) 
    {
        for (const size_t idx : CameraModel::extra_params_idxs) {
            if (std::abs(params[idx]) > max_extra_param) {
                return true;
            }
        }
        return false;
    }

    template <typename T>
    static inline T CamFromImgThreshold(const T* params, T threshold)
    {
        T mean_focal_length = 0;
        for (const size_t idx : CameraModel::focal_length_idxs) {
            mean_focal_length += params[idx];
        }
        mean_focal_length /= CameraModel::focal_length_idxs.size();
        return threshold / mean_focal_length;
    }

    static inline void IterativeUndistortion(const double* params,
        double* u,
        double* v)
    {
        // Parameters for Newton iteration. 100 iterations should be enough for
        // complex camera models with higher order terms.
        constexpr size_t kNumIterations = 100;
        constexpr double kMaxStepNorm = 1e-10;
        // Trust region: step_x.norm() <= max(x.norm() * kRelStepRadius, kStepRadius)
        constexpr double kRelStepRadius = 0.1;
        constexpr double kStepRadius = 0.1;

        Eigen::Matrix2d J;
        const Eigen::Vector2d x0(*u, *v);
        Eigen::Vector2d x(*u, *v);
        Eigen::Vector2d dx;

        ceres::Jet<double, 2> params_jet[CameraModel::num_extra_params];
        for (size_t i = 0; i < CameraModel::num_extra_params; ++i) {
            params_jet[i] = ceres::Jet<double, 2>(params[i]);
        }
        for (size_t i = 0; i < kNumIterations; ++i) {
            // Get Jacobian
            ceres::Jet<double, 2> x_jet[2];
            x_jet[0] = ceres::Jet<double, 2>(x(0), 0);
            x_jet[1] = ceres::Jet<double, 2>(x(1), 1);
            ceres::Jet<double, 2> dx_jet[2];
            CameraModel::Distortion(
                params_jet, x_jet[0], x_jet[1], &dx_jet[0], &dx_jet[1]);
            dx[0] = dx_jet[0].a;
            dx[1] = dx_jet[1].a;
            J(0, 0) = dx_jet[0].v[0] + 1;
            J(0, 1) = dx_jet[0].v[1];
            J(1, 0) = dx_jet[1].v[0];
            J(1, 1) = dx_jet[1].v[1] + 1;

            // Update
            Eigen::Vector2d step_x = J.partialPivLu().solve(x + dx - x0);
            const double radius_sqr =
                std::max(x.squaredNorm() * kRelStepRadius * kRelStepRadius,
                    kStepRadius * kStepRadius);
            const double step_x_norm_sqr = step_x.squaredNorm();
            if (step_x_norm_sqr > radius_sqr) {
                step_x *= std::sqrt(radius_sqr / step_x_norm_sqr);
            }
            x -= step_x;
            if (step_x.squaredNorm() < kMaxStepNorm) {
                break;
            }
        }

        *u = x(0);
        *v = x(1);
    }

};

#ifndef CAMERA_MODEL_DEFINITIONS
#define CAMERA_MODEL_DEFINITIONS(model_id_val,                                \
                                 model_name_val,                              \
                                 num_focal_params_val,                        \
                                 num_pp_params_val,                           \
                                 num_extra_params_val)                        \
  static constexpr size_t num_params =                                        \
      (num_focal_params_val) + (num_pp_params_val) + (num_extra_params_val);  \
  static constexpr size_t num_focal_params = num_focal_params_val;            \
  static constexpr size_t num_pp_params = num_pp_params_val;                  \
  static constexpr size_t num_extra_params = num_extra_params_val;            \
  static constexpr CameraModelId model_id = model_id_val;                     \
  static const std::string model_name;                                        \
  static const std::string params_info;                                       \
  static const std::array<size_t, (num_focal_params_val)> focal_length_idxs;  \
  static const std::array<size_t, (num_pp_params_val)> principal_point_idxs;  \
  static const std::array<size_t, (num_extra_params_val)> extra_params_idxs;  \
                                                                              \
  static inline CameraModelId InitializeModelId() { return model_id_val; };   \
  static inline std::string InitializeModelName() { return model_name_val; }; \
  static inline std::string InitializeParamsInfo();                           \
  static inline std::array<size_t, (num_focal_params_val)>                    \
  InitializeFocalLengthIdxs();                                                \
  static inline std::array<size_t, (num_pp_params_val)>                       \
  InitializePrincipalPointIdxs();                                             \
  static inline std::array<size_t, (num_extra_params_val)>                    \
  InitializeExtraParamsIdxs();                                                \
                                                                              \
  static inline std::vector<double> InitializeParams(                         \
      double focal_length, size_t width, size_t height);                      \
  template <typename T>                                                       \
  static void ImgFromCam(const T* params, T u, T v, T w, T* x, T* y);         \
  static inline void CamFromImg(const double* params,                         \
                                double x,                                     \
                                double y,                                     \
                                double* u,                                    \
                                double* v,                                    \
                                double* w);                                   \
  template <typename T>                                                       \
  static void Distortion(const T* extra_params, T u, T v, T* du, T* dv);
#endif

// Simple Pinhole camera model.
// No Distortion is assumed. Only focal length and principal point is modeled.
// Parameter list is expected in the following order:
//   f, cx, cy
// See https://en.wikipedia.org/wiki/Pinhole_camera_model
struct SimplePinholeCameraModel
    : public BaseCameraModel<SimplePinholeCameraModel> {
    CAMERA_MODEL_DEFINITIONS(
        CameraModelId::kSimplePinhole, "SIMPLE_PINHOLE", 1, 2, 0)
};
// Pinhole camera model.
// No Distortion is assumed. Only focal length and principal point is modeled.
// Parameter list is expected in the following order:
//    fx, fy, cx, cy
// See https://en.wikipedia.org/wiki/Pinhole_camera_model
struct PinholeCameraModel : public BaseCameraModel<PinholeCameraModel> {
    CAMERA_MODEL_DEFINITIONS(CameraModelId::kPinhole, "PINHOLE", 2, 2, 0)
};
// Simple camera model with one focal length and one radial distortion parameter.
// This model is similar to the camera model that VisualSfM uses with the
// difference that the distortion here is applied to the projections and
// not to the measurements.
// Parameter list is expected in the following order:
//    f, cx, cy, k
struct SimpleRadialCameraModel
    : public BaseCameraModel<SimpleRadialCameraModel> {
    CAMERA_MODEL_DEFINITIONS(
        CameraModelId::kSimpleRadial, "SIMPLE_RADIAL", 1, 2, 1)
};
// Simple camera model with one focal length and two radial distortion parameters.
// This model is equivalent to the camera model that Bundler uses
// (except for an inverse z-axis in the camera coordinate system).
// Parameter list is expected in the following order:
//    f, cx, cy, k1, k2
struct RadialCameraModel : public BaseCameraModel<RadialCameraModel> {
    CAMERA_MODEL_DEFINITIONS(CameraModelId::kRadial, "RADIAL", 1, 2, 2)
};

// OpenCV camera model.
//
// Based on the pinhole camera model. Additionally models radial and
// tangential distortion (up to 2nd degree of coefficients). Not suitable for
// large radial distortions of fish-eye cameras.
//
// Parameter list is expected in the following order:
//
//    fx, fy, cx, cy, k1, k2, p1, p2
//
// See
// http://docs.opencv.org/modules/calib3d/doc/camera_calibration_and_3d_reconstruction.html
struct OpenCVCameraModel : public BaseCameraModel<OpenCVCameraModel> {
    CAMERA_MODEL_DEFINITIONS(CameraModelId::kOpenCV, "OPENCV", 2, 2, 4)
};



std::vector<double> CameraModelInitializeParams(CameraModelId model_id,
    double focal_length,
    size_t width,
    size_t height);


#define CAMERA_STRUCT_DEFINE(CameraModel)                    \
  constexpr CameraModelId CameraModel::model_id;          \
  const std::string CameraModel::model_name =             \
      CameraModel::InitializeModelName();                 \
  constexpr size_t CameraModel::num_params;               \
  const std::string CameraModel::params_info =            \
      CameraModel::InitializeParamsInfo();                \
  const std::array<size_t, CameraModel::num_focal_params> \
      CameraModel::focal_length_idxs =                    \
          CameraModel::InitializeFocalLengthIdxs();       \
  const std::array<size_t, CameraModel::num_pp_params>    \
      CameraModel::principal_point_idxs =                 \
          CameraModel::InitializePrincipalPointIdxs();    \
  const std::array<size_t, CameraModel::num_extra_params> \
      CameraModel::extra_params_idxs =                    \
          CameraModel::InitializeExtraParamsIdxs();



const std::string& CameraModelParamsInfo(CameraModelId model_id);

struct Camera {
    // The unique identifier of the camera.
    camera_t camera_id = kInvalidCameraId;

    // The identifier of the camera model.
    CameraModelId model_id = CameraModelId::kInvalid;

    // The dimensions of the image, 0 if not initialized.
    size_t width = 0;
    size_t height = 0;

    // The focal length, principal point, and extra parameters. If the camera
    // model is not specified, this vector is empty.
    std::vector<double> params;

    // Whether there is a safe prior for the focal length,
    // e.g. manually provided or extracted from EXIF
    bool has_prior_focal_length = false;

    // Initialize parameters for given camera model and focal length, and set
    // the principal point to be the image center.
    static Camera CreateFromModelId(camera_t camera_id,
        const CameraModelId&model_id,
        const double&focal_length,
        const size_t&width,
        const size_t&height);
    static Camera CreateFromModelName(camera_t camera_id,
        const std::string& model_name,
        const double&focal_length,
        const size_t&width,
        const size_t&height);

    inline const std::string& ModelName() const;

    // Access focal length parameters.
    double MeanFocalLength() const;
    inline double FocalLength() const;
    inline double FocalLengthX() const;
    inline double FocalLengthY() const;
    inline void SetFocalLength(double f);
    inline void SetFocalLengthX(double fx);
    inline void SetFocalLengthY(double fy);

    // Access principal point parameters. Only works if there are two
    // principal point parameters.
    inline double PrincipalPointX() const;
    inline double PrincipalPointY() const;
    inline void SetPrincipalPointX(double cx);
    inline void SetPrincipalPointY(double cy);

    // Get the indices of the parameter groups in the parameter vector.
    inline span<const size_t> FocalLengthIdxs() const;
    inline span<const size_t> PrincipalPointIdxs() const;
    inline span<const size_t> ExtraParamsIdxs() const;

    // Get intrinsic calibration matrix composed from focal length and principal
    // point parameters, excluding distortion parameters.
    Eigen::Matrix3d CalibrationMatrix() const {
        Eigen::Matrix3d K = Eigen::Matrix3d::Identity();

        const span<const size_t> idxs = FocalLengthIdxs();
        if (idxs.size() == 1) {
            K(0, 0) = params[idxs[0]];
            K(1, 1) = params[idxs[0]];
        }
        else if (idxs.size() == 2) {
            K(0, 0) = params[idxs[0]];
            K(1, 1) = params[idxs[1]];
        }
        else {
            LOG_ERR_OUT << "Camera model must either have 1 or 2 focal length parameters.";
        }

        K(0, 2) = PrincipalPointX();
        K(1, 2) = PrincipalPointY();

        return K;
    }

    // Get human-readable information about the parameter vector ordering.
    inline const std::string& ParamsInfo() const
    {
        return CameraModelParamsInfo(model_id);
    }

    // Concatenate parameters as comma-separated list.
    std::string ParamsToString() const;

    // Set camera parameters from comma-separated list.
    bool SetParamsFromString(const std::string& string);

    // Check whether parameters are valid, i.e. the parameter vector has
    // the correct dimensions that match the specified camera model.
    inline bool VerifyParams() const;

    // Check whether camera is already undistorted.
    bool IsUndistorted() const;

    // Check whether camera has bogus parameters.
    inline bool HasBogusParams(double min_focal_length_ratio,
        double max_focal_length_ratio,
        double max_extra_param) const;

    // Project point in image plane to world / infinity.
    inline Eigen::Vector2d CamFromImg(const Eigen::Vector2d& image_point) const;

    // Convert pixel threshold in image plane to camera frame.
    inline double CamFromImgThreshold(double threshold) const;

    // Project point from camera frame to image plane.
    inline Eigen::Vector2d ImgFromCam(const Eigen::Vector2d& cam_point) const;

    // Rescale camera dimensions and accordingly the focal length and
    // and the principal point.
    void Rescale(double scale);
    void Rescale(size_t new_width, size_t new_height);

    inline bool operator==(const Camera& other) const;
    inline bool operator!=(const Camera& other) const;
    inline bool operator<(const Camera& other) const;
};

const std::string& CameraModelIdToName(const CameraModelId model_id);
std::ostream& operator<<(std::ostream& stream, const Camera& camera);


std::vector<double> CameraModelInitializeParams(CameraModelId model_id,
    double focal_length,
    size_t width,
    size_t height);
span<const size_t> CameraModelFocalLengthIdxs(CameraModelId model_id);
span<const size_t> CameraModelPrincipalPointIdxs(CameraModelId model_id);
span<const size_t> CameraModelExtraParamsIdxs(CameraModelId model_id);
size_t CameraModelNumParams(CameraModelId model_id);
bool CameraModelVerifyParams(CameraModelId model_id,
    const std::vector<double>& params);
bool CameraModelHasBogusParams(CameraModelId model_id,
    const std::vector<double>& params,
    size_t width,
    size_t height,
    double min_focal_length_ratio,
    double max_focal_length_ratio,
    double max_extra_param);
bool CameraModelVerifyParams(CameraModelId model_id, const std::vector<double>& params);
bool CameraModelHasBogusParams(CameraModelId model_id,
    const std::vector<double>& params,
    size_t width,
    size_t height,
    double min_focal_length_ratio,
    double max_focal_length_ratio,
    double max_extra_param);
inline Eigen::Vector2d CameraModelImgFromCam(CameraModelId model_id,
    const std::vector<double>& params,
    const Eigen::Vector3d& uvw);
inline Eigen::Vector3d CameraModelCamFromImg(CameraModelId model_id,
    const std::vector<double>& params,
    const Eigen::Vector2d& xy);
inline double CameraModelCamFromImgThreshold(CameraModelId model_id,
    const std::vector<double>& params,
    double threshold);
const std::string& Camera::ModelName() const
{
    return CameraModelIdToName(model_id);
}
double Camera::FocalLength() const {
    const span<const size_t> idxs = FocalLengthIdxs();
    return params[idxs[0]];
}

double Camera::FocalLengthX() const {
    const span<const size_t> idxs = FocalLengthIdxs();
    return params[idxs[0]];
}

double Camera::FocalLengthY() const {
    const span<const size_t> idxs = FocalLengthIdxs();
    return params[idxs[(idxs.size() == 1) ? 0 : 1]];
}

void Camera::SetFocalLength(const double f) {
    const span<const size_t> idxs = FocalLengthIdxs();
    for (const size_t idx : idxs) {
        params[idx] = f;
    }
}

void Camera::SetFocalLengthX(const double fx) {
    const span<const size_t> idxs = FocalLengthIdxs();
    params[idxs[0]] = fx;
}

void Camera::SetFocalLengthY(const double fy) {
    const span<const size_t> idxs = FocalLengthIdxs();
    params[idxs[1]] = fy;
}

double Camera::PrincipalPointX() const {
    const span<const size_t> idxs = PrincipalPointIdxs();
    return params[idxs[0]];
}

double Camera::PrincipalPointY() const {
    const span<const size_t> idxs = PrincipalPointIdxs();
    return params[idxs[1]];
}

void Camera::SetPrincipalPointX(const double cx) {
    const span<const size_t> idxs = PrincipalPointIdxs();
    params[idxs[0]] = cx;
}

void Camera::SetPrincipalPointY(const double cy) {
    const span<const size_t> idxs = PrincipalPointIdxs();
    params[idxs[1]] = cy;
}

span<const size_t> Camera::FocalLengthIdxs() const {
    return CameraModelFocalLengthIdxs(model_id);
}

span<const size_t> Camera::PrincipalPointIdxs() const {
    return CameraModelPrincipalPointIdxs(model_id);
}

span<const size_t> Camera::ExtraParamsIdxs() const {
    return CameraModelExtraParamsIdxs(model_id);
}

bool Camera::VerifyParams() const {
    return CameraModelVerifyParams(model_id, params);
}

bool Camera::HasBogusParams(const double min_focal_length_ratio,
    const double max_focal_length_ratio,
    const double max_extra_param) const {
    return CameraModelHasBogusParams(model_id,
        params,
        width,
        height,
        min_focal_length_ratio,
        max_focal_length_ratio,
        max_extra_param);
}

Eigen::Vector2d Camera::CamFromImg(const Eigen::Vector2d& image_point) const {
    return CameraModelCamFromImg(model_id, params, image_point).hnormalized();
}

double Camera::CamFromImgThreshold(const double threshold) const {
    return CameraModelCamFromImgThreshold(model_id, params, threshold);
}

Eigen::Vector2d Camera::ImgFromCam(const Eigen::Vector2d& cam_point) const {
    return CameraModelImgFromCam(model_id, params, cam_point.homogeneous());
}

bool Camera::operator==(const Camera& other) const {
    return camera_id == other.camera_id && model_id == other.model_id &&
        width == other.width && height == other.height &&
        params == other.params &&
        has_prior_focal_length == other.has_prior_focal_length;
}

bool Camera::operator!=(const Camera& other) const { return !(*this == other); }
bool Camera::operator<(const Camera& other) const { return this->camera_id< other.camera_id; }



////////////////////////////////////////////////////////////////////////////////
// SimplePinholeCameraModel

std::string SimplePinholeCameraModel::InitializeParamsInfo() {
    return "f, cx, cy";
}

std::array<size_t, 1> SimplePinholeCameraModel::InitializeFocalLengthIdxs() {
    return { 0 };
}

std::array<size_t, 2> SimplePinholeCameraModel::InitializePrincipalPointIdxs() {
    return { 1, 2 };
}

std::array<size_t, 0> SimplePinholeCameraModel::InitializeExtraParamsIdxs() {
    return {};
}

std::vector<double> SimplePinholeCameraModel::InitializeParams(
    const double focal_length, const size_t width, const size_t height) {
    return { focal_length, width / 2.0, height / 2.0 };
}

template <typename T>
void SimplePinholeCameraModel::ImgFromCam(
    const T* params, T u, T v, T w, T* x, T* y) {
    const T f = params[0];
    const T c1 = params[1];
    const T c2 = params[2];

    // No Distortion

    // Transform to image coordinates
    *x = f * u / w + c1;
    *y = f * v / w + c2;
}

void SimplePinholeCameraModel::CamFromImg(
    const double* params, double x, double y, double* u, double* v, double* w) {
    const double f = params[0];
    const double c1 = params[1];
    const double c2 = params[2];

    *u = (x - c1) / f;
    *v = (y - c2) / f;
    *w = 1;
}

////////////////////////////////////////////////////////////////////////////////
// PinholeCameraModel

std::string PinholeCameraModel::InitializeParamsInfo() {
    return "fx, fy, cx, cy";
}

std::array<size_t, 2> PinholeCameraModel::InitializeFocalLengthIdxs() {
    return { 0, 1 };
}

std::array<size_t, 2> PinholeCameraModel::InitializePrincipalPointIdxs() {
    return { 2, 3 };
}

std::array<size_t, 0> PinholeCameraModel::InitializeExtraParamsIdxs() {
    return {};
}

std::vector<double> PinholeCameraModel::InitializeParams(
    const double focal_length, const size_t width, const size_t height) {
    return { focal_length, focal_length, width / 2.0, height / 2.0 };
}

template <typename T>
void PinholeCameraModel::ImgFromCam(
    const T* params, T u, T v, T w, T* x, T* y) {
    const T f1 = params[0];
    const T f2 = params[1];
    const T c1 = params[2];
    const T c2 = params[3];

    // No Distortion

    // Transform to image coordinates
    *x = f1 * u / w + c1;
    *y = f2 * v / w + c2;
}

void PinholeCameraModel::CamFromImg(
    const double* params, double x, double y, double* u, double* v, double* w) {
    const double f1 = params[0];
    const double f2 = params[1];
    const double c1 = params[2];
    const double c2 = params[3];

    *u = (x - c1) / f1;
    *v = (y - c2) / f2;
    *w = 1;
}

////////////////////////////////////////////////////////////////////////////////
// SimpleRadialCameraModel

std::string SimpleRadialCameraModel::InitializeParamsInfo() {
    return "f, cx, cy, k";
}

std::array<size_t, 1> SimpleRadialCameraModel::InitializeFocalLengthIdxs() {
    return { 0 };
}

std::array<size_t, 2> SimpleRadialCameraModel::InitializePrincipalPointIdxs() {
    return { 1, 2 };
}

std::array<size_t, 1> SimpleRadialCameraModel::InitializeExtraParamsIdxs() {
    return { 3 };
}

std::vector<double> SimpleRadialCameraModel::InitializeParams(
    const double focal_length, const size_t width, const size_t height) {
    return { focal_length, width / 2.0, height / 2.0, 0 };
}

template <typename T>
void SimpleRadialCameraModel::ImgFromCam(
    const T* params, T u, T v, T w, T* x, T* y) {
    const T f = params[0];
    const T c1 = params[1];
    const T c2 = params[2];

    u /= w;
    v /= w;

    // Distortion
    T du, dv;
    Distortion(&params[3], u, v, &du, &dv);
    *x = u + du;
    *y = v + dv;

    // Transform to image coordinates
    *x = f * *x + c1;
    *y = f * *y + c2;
}

void SimpleRadialCameraModel::CamFromImg(
    const double* params, double x, double y, double* u, double* v, double* w) {
    const double f = params[0];
    const double c1 = params[1];
    const double c2 = params[2];

    // Lift points to normalized plane
    *u = (x - c1) / f;
    *v = (y - c2) / f;
    *w = 1;

    IterativeUndistortion(&params[3], u, v);
}

template <typename T>
void SimpleRadialCameraModel::Distortion(
    const T* extra_params, const T u, const T v, T* du, T* dv) {
    const T k = extra_params[0];

    const T u2 = u * u;
    const T v2 = v * v;
    const T r2 = u2 + v2;
    const T radial = k * r2;
    *du = u * radial;
    *dv = v * radial;
}

////////////////////////////////////////////////////////////////////////////////
// RadialCameraModel

std::string RadialCameraModel::InitializeParamsInfo() {
    return "f, cx, cy, k1, k2";
}

std::array<size_t, 1> RadialCameraModel::InitializeFocalLengthIdxs() {
    return { 0 };
}

std::array<size_t, 2> RadialCameraModel::InitializePrincipalPointIdxs() {
    return { 1, 2 };
}

std::array<size_t, 2> RadialCameraModel::InitializeExtraParamsIdxs() {
    return { 3, 4 };
}

std::vector<double> RadialCameraModel::InitializeParams(
    const double focal_length, const size_t width, const size_t height) {
    return { focal_length, width / 2.0, height / 2.0, 0, 0 };
}

template <typename T>
void RadialCameraModel::ImgFromCam(const T* params, T u, T v, T w, T* x, T* y) {
    const T f = params[0];
    const T c1 = params[1];
    const T c2 = params[2];

    u /= w;
    v /= w;

    // Distortion
    T du, dv;
    Distortion(&params[3], u, v, &du, &dv);
    *x = u + du;
    *y = v + dv;

    // Transform to image coordinates
    *x = f * *x + c1;
    *y = f * *y + c2;
}

void RadialCameraModel::CamFromImg(
    const double* params, double x, double y, double* u, double* v, double* w) {
    const double f = params[0];
    const double c1 = params[1];
    const double c2 = params[2];

    // Lift points to normalized plane
    *u = (x - c1) / f;
    *v = (y - c2) / f;
    *w = 1;

    IterativeUndistortion(&params[3], u, v);
}

template <typename T>
void RadialCameraModel::Distortion(
    const T* extra_params, const T u, const T v, T* du, T* dv) {
    const T k1 = extra_params[0];
    const T k2 = extra_params[1];

    const T u2 = u * u;
    const T v2 = v * v;
    const T r2 = u2 + v2;
    const T radial = k1 * r2 + k2 * r2 * r2;
    *du = u * radial;
    *dv = v * radial;
}

std::string OpenCVCameraModel::InitializeParamsInfo() {
    return "fx, fy, cx, cy, k1, k2, p1, p2";
}

std::array<size_t, 2> OpenCVCameraModel::InitializeFocalLengthIdxs() {
    return { 0, 1 };
}

std::array<size_t, 2> OpenCVCameraModel::InitializePrincipalPointIdxs() {
    return { 2, 3 };
}

std::array<size_t, 4> OpenCVCameraModel::InitializeExtraParamsIdxs() {
    return { 4, 5, 6, 7 };
}

std::vector<double> OpenCVCameraModel::InitializeParams(
    const double focal_length, const size_t width, const size_t height) {
    return { focal_length, focal_length, width / 2.0, height / 2.0, 0, 0, 0, 0 };
}

template <typename T>
void OpenCVCameraModel::ImgFromCam(const T* params, T u, T v, T w, T* x, T* y) {
    const T f1 = params[0];
    const T f2 = params[1];
    const T c1 = params[2];
    const T c2 = params[3];

    u /= w;
    v /= w;

    // Distortion
    T du, dv;
    Distortion(&params[4], u, v, &du, &dv);
    *x = u + du;
    *y = v + dv;

    // Transform to image coordinates
    *x = f1 * *x + c1;
    *y = f2 * *y + c2;
}

void OpenCVCameraModel::CamFromImg(
    const double* params, double x, double y, double* u, double* v, double* w) {
    const double f1 = params[0];
    const double f2 = params[1];
    const double c1 = params[2];
    const double c2 = params[3];

    // Lift points to normalized plane
    *u = (x - c1) / f1;
    *v = (y - c2) / f2;
    *w = 1;

    IterativeUndistortion(&params[4], u, v);
}

template <typename T>
void OpenCVCameraModel::Distortion(
    const T* extra_params, const T u, const T v, T* du, T* dv) {
    const T k1 = extra_params[0];
    const T k2 = extra_params[1];
    const T p1 = extra_params[2];
    const T p2 = extra_params[3];

    const T u2 = u * u;
    const T uv = u * v;
    const T v2 = v * v;
    const T r2 = u2 + v2;
    const T radial = k1 * r2 + k2 * r2 * r2;
    *du = u * radial + T(2) * p1 * uv + p2 * (r2 + T(2) * u2);
    *dv = v * radial + T(2) * p2 * uv + p1 * (r2 + T(2) * v2);
}

Eigen::Vector2d CameraModelImgFromCam(const CameraModelId model_id,
    const std::vector<double>& params,
    const Eigen::Vector3d& uvw) {
    Eigen::Vector2d xy;
    switch (model_id) {
#define CAMERA_MODEL_CASE(CameraModel)                               \
  case CameraModel::model_id:                                        \
    CameraModel::ImgFromCam(                                         \
        params.data(), uvw.x(), uvw.y(), uvw.z(), &xy.x(), &xy.y()); \
    break; 
        CAMERA_MODEL_SWITCH_CASES 
#undef CAMERA_MODEL_CASE
    }
    return xy;
}

Eigen::Vector3d CameraModelCamFromImg(const CameraModelId model_id,
    const std::vector<double>& params,
    const Eigen::Vector2d& xy) {
    Eigen::Vector3d uvw;
    switch (model_id) {
#define CAMERA_MODEL_CASE(CameraModel)                                \
  case CameraModel::model_id:                                         \
    CameraModel::CamFromImg(                                          \
        params.data(), xy.x(), xy.y(), &uvw.x(), &uvw.y(), &uvw.z()); \
    break;

        CAMERA_MODEL_SWITCH_CASES

#undef CAMERA_MODEL_CASE
    }
    return uvw;
}

double CameraModelCamFromImgThreshold(const CameraModelId model_id,
    const std::vector<double>& params,
    const double threshold) {
    switch (model_id) {
#define CAMERA_MODEL_CASE(CameraModel)                                 \
  case CameraModel::model_id:                                          \
    return CameraModel::CamFromImgThreshold(params.data(), threshold); \
    break;

        CAMERA_MODEL_SWITCH_CASES

#undef CAMERA_MODEL_CASE
    }

    return -1;
}

#endif // _CAMERA_H_
