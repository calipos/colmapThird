#include "camera.h"


#define CAMERA_MODEL_CASE(CameraModel)                    \
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
CAMERA_MODEL_CASES
#undef CAMERA_MODEL_CASE


std::unordered_map<std::string, CameraModelId> InitialzeCameraModelNameToId() {
    std::unordered_map<std::string, CameraModelId> camera_model_name_to_id;

#define CAMERA_MODEL_CASE(CameraModel)                     \
  camera_model_name_to_id.emplace(CameraModel::model_name, \
                                  CameraModel::model_id);
    CAMERA_MODEL_CASES
#undef CAMERA_MODEL_CASE
        return camera_model_name_to_id;
}

std::unordered_map<CameraModelId, const std::string*>
InitialzeCameraModelIdToName() {
    std::unordered_map<CameraModelId, const std::string*> camera_model_id_to_name;
#define CAMERA_MODEL_CASE(CameraModel)                   \
  camera_model_id_to_name.emplace(CameraModel::model_id, \
                                  &CameraModel::model_name);
    CAMERA_MODEL_CASES
#undef CAMERA_MODEL_CASE
        return camera_model_id_to_name;
}

static const std::unordered_map<std::string, CameraModelId>
kCameraModelNameToId = InitialzeCameraModelNameToId();

static const std::unordered_map<CameraModelId, const std::string*>
kCameraModelIdToName = InitialzeCameraModelIdToName();

bool ExistsCameraModelWithName(const std::string & model_name) {
    return kCameraModelNameToId.count(model_name) > 0;
}

bool ExistsCameraModelWithId(const CameraModelId model_id) {
    return kCameraModelIdToName.count(model_id) > 0;
}

CameraModelId CameraModelNameToId(const std::string & model_name) {
    const auto it = kCameraModelNameToId.find(model_name);
    if (it == kCameraModelNameToId.end()) {
        return CameraModelId::kInvalid;
    }
    else {
        return it->second;
    }
}

const std::string& CameraModelIdToName(const CameraModelId model_id) {
    const auto it = kCameraModelIdToName.find(model_id);
    if (it == kCameraModelIdToName.end()) {
        const static std::string kEmptyModelName = "";
        return kEmptyModelName;
    }
    else {
        return *(it->second);
    }
}

std::vector<double> CameraModelInitializeParams(const CameraModelId model_id,
    const double focal_length,
    const size_t width,
    const size_t height) {
    switch (model_id) {
#define CAMERA_MODEL_CASE(CameraModel)                                 \
  case CameraModel::model_id:                                          \
    return CameraModel::InitializeParams(focal_length, width, height); \
    break;
        CAMERA_MODEL_SWITCH_CASES
#undef CAMERA_MODEL_CASE
    }
}

const std::string& CameraModelParamsInfo(const CameraModelId model_id) {
    switch (model_id) {
#define CAMERA_MODEL_CASE(CameraModel) \
  case CameraModel::model_id:          \
    return CameraModel::params_info;   \
    break;
        CAMERA_MODEL_SWITCH_CASES
#undef CAMERA_MODEL_CASE
    }
    const static std::string kEmptyParamsInfo = "";
    return kEmptyParamsInfo;
}

span<const size_t> CameraModelFocalLengthIdxs(const CameraModelId model_id) {
    switch (model_id) {
#define CAMERA_MODEL_CASE(CameraModel)              \
  case CameraModel::model_id:                       \
    return {CameraModel::focal_length_idxs.data(),  \
            CameraModel::focal_length_idxs.size()}; \
    break;
        CAMERA_MODEL_SWITCH_CASES
#undef CAMERA_MODEL_CASE
    }
    return { nullptr, 0 };
}

span<const size_t> CameraModelPrincipalPointIdxs(const CameraModelId model_id) {
    switch (model_id) {
#define CAMERA_MODEL_CASE(CameraModel)                 \
  case CameraModel::model_id:                          \
    return {CameraModel::principal_point_idxs.data(),  \
            CameraModel::principal_point_idxs.size()}; \
    break;
        CAMERA_MODEL_SWITCH_CASES
#undef CAMERA_MODEL_CASE
    }
    return { nullptr, 0 };
}

span<const size_t> CameraModelExtraParamsIdxs(const CameraModelId model_id) {
    switch (model_id) {
#define CAMERA_MODEL_CASE(CameraModel)              \
  case CameraModel::model_id:                       \
    return {CameraModel::extra_params_idxs.data(),  \
            CameraModel::extra_params_idxs.size()}; \
    break;
        CAMERA_MODEL_SWITCH_CASES
#undef CAMERA_MODEL_CASE
    }
    return { nullptr, 0 };
}

size_t CameraModelNumParams(const CameraModelId model_id) {
    switch (model_id) {
#define CAMERA_MODEL_CASE(CameraModel) \
  case CameraModel::model_id:          \
    return CameraModel::num_params;

        CAMERA_MODEL_SWITCH_CASES

#undef CAMERA_MODEL_CASE
    }

    return 0;
}

bool CameraModelVerifyParams(const CameraModelId model_id,
    const std::vector<double>&params) {
    switch (model_id) {
#define CAMERA_MODEL_CASE(CameraModel)              \
  case CameraModel::model_id:                       \
    if (params.size() == CameraModel::num_params) { \
      return true;                                  \
    }                                               \
    break;

        CAMERA_MODEL_SWITCH_CASES

#undef CAMERA_MODEL_CASE
    }

    return false;
}

bool CameraModelHasBogusParams(const CameraModelId model_id,
    const std::vector<double>&params,
    const size_t width,
    const size_t height,
    const double min_focal_length_ratio,
    const double max_focal_length_ratio,
    const double max_extra_param) {
    switch (model_id) {
#define CAMERA_MODEL_CASE(CameraModel)                         \
  case CameraModel::model_id:                                  \
    return CameraModel::HasBogusParams(params,                 \
                                       width,                  \
                                       height,                 \
                                       min_focal_length_ratio, \
                                       max_focal_length_ratio, \
                                       max_extra_param);       \
    break;

        CAMERA_MODEL_SWITCH_CASES

#undef CAMERA_MODEL_CASE
    }

    return false;
}



Camera Camera::CreateFromModelId(camera_t camera_id,
    const CameraModelId&model_id,
    const double&focal_length,
    const size_t&width,
    const size_t&height) {
    Camera camera;
    camera.camera_id = camera_id;
    camera.model_id = model_id;
    camera.width = width;
    camera.height = height;
    camera.params = CameraModelInitializeParams(model_id, focal_length, width, height);
    return camera;
}
Camera Camera::CreateFromModelName(camera_t camera_id,
    const std::string& model_name,
    const double&focal_length,
    const size_t&width,
    const size_t&height) {
    return CreateFromModelId(
        camera_id, enumStr2Value<CameraModelId>(model_name.c_str()), focal_length, width, height);
}