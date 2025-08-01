#include <unordered_set>
#include "types.h"
#include "camera_rig.h"
#include "log.h"
#include "misc.h"
CameraRig::CameraRig() {}

size_t CameraRig::NumCameras() const { return cams_from_rigs_.size(); }

size_t CameraRig::NumSnapshots() const { return snapshots_.size(); }

bool CameraRig::HasCamera(const camera_t&camera_id) const {
    return cams_from_rigs_.count(camera_id);
}

camera_t CameraRig::RefCameraId() const { return ref_camera_id_; }

void CameraRig::SetRefCameraId(const camera_t camera_id) {
    if (!HasCamera(camera_id))
    {
        LOG_ERR_OUT << "!HasCamera";
        return;
    }
    ref_camera_id_ = camera_id;
}

std::vector<camera_t> CameraRig::GetCameraIds() const {
    std::vector<camera_t> rig_camera_ids;
    rig_camera_ids.reserve(cams_from_rigs_.size());
    for (const auto& rig_camera : cams_from_rigs_) {
        rig_camera_ids.push_back(rig_camera.first);
    }
    return rig_camera_ids;
}

const std::vector<std::vector<image_t>>& CameraRig::Snapshots() const {
    return snapshots_;
}

void CameraRig::AddCamera(const camera_t camera_id,
    const Rigid3d& cam_from_rig) {
    if (!HasCamera(camera_id))
    {
        LOG_ERR_OUT << "!HasCamera";
        return;
    }
    if (NumSnapshots()!=0)
    {
        LOG_ERR_OUT << "NumSnapshots()!=0";
        return;
    }
    cams_from_rigs_.emplace(camera_id, cam_from_rig);
}

void CameraRig::AddSnapshot(const std::vector<image_t>& image_ids) {
    if (image_ids.size()==0)
    {
        LOG_ERR_OUT << "image_ids.size()==0";
        return;
    }
    if (image_ids.size() > NumCameras())
    {
        LOG_ERR_OUT << "image_ids.size() > NumCameras()";
        return;
    } 
    if (true==VectorContainsDuplicateValues(image_ids))
    {
        LOG_ERR_OUT << "image_ids.size() > NumCameras()";
        return;
    }
    //THROW_CHECK(!VectorContainsDuplicateValues(image_ids));
    snapshots_.push_back(image_ids);
}

void CameraRig::Check(const std::vector<struct Camera>& cameraList,
    std::vector<class Image>& imageList) const {
    if (!HasCamera(ref_camera_id_))
    {
        LOG_ERR_OUT << "!ref_camera_id_";
        return;
    }
    for (const auto& rig_camera : cams_from_rigs_) {
        if (rig_camera.first < 0 || rig_camera.first >= cameraList.size())
        {
            LOG_ERR_OUT << "rig_camera.first < 0 || rig_camera.first >= cameraList.size()";
            return;
        }
    }

    std::unordered_set<image_t> all_image_ids;
    for (const auto& snapshot : snapshots_) {
        if (snapshot.size() == 0)
        {
            LOG_ERR_OUT << "snapshot.size()==0";
            return;
        }
        if (snapshot.size() > NumCameras())
        {
            LOG_ERR_OUT << "image_ids.size() > NumCameras()";
            return;
        }
        bool has_ref_camera = false;
        for (const auto image_id : snapshot) {
            if (image_id < 0 || image_id >= imageList.size())
            {
                LOG_ERR_OUT << "image_id < 0 || image_id >= imageList.size()";
                return;
            }
            if (all_image_ids.count(image_id) != 0)
            {
                LOG_ERR_OUT << "all_image_ids.count(image_id)!= 0";
                return;
            }
            all_image_ids.insert(image_id);
            const auto& image = imageList[image_id];
            if (!HasCamera(image.CameraId()))
            {
                LOG_ERR_OUT << "!HasCamera(image.CameraId())";
                return;
            }
            if (image.CameraId() == ref_camera_id_) {
                has_ref_camera = true;
            }
        }
        if (!has_ref_camera)
        {
            LOG_ERR_OUT << "!has_ref_camera";
            return;
        }
    }
}

const Rigid3d& CameraRig::CamFromRig(const camera_t camera_id) const {
    return cams_from_rigs_.at(camera_id);
}

Rigid3d& CameraRig::CamFromRig(const camera_t camera_id) {
    return cams_from_rigs_.at(camera_id);
}

double CameraRig::ComputeRigFromWorldScale(
    const Reconstruction& reconstruction) const {
    if (NumSnapshots() < 0)
    {
        LOG_ERR_OUT << "NumSnapshots()< 0";
        return 0;
    }
    const size_t num_cameras = NumCameras();
    if (num_cameras==0)
    {
        LOG_ERR_OUT << "num_cameras==0";
        return 0;
    }
    double rig_from_world_scale = 0;
    size_t num_dists = 0;
    std::vector<Eigen::Vector3d> proj_centers_in_rig(num_cameras);
    std::vector<Eigen::Vector3d> proj_centers_in_world(num_cameras);
    for (const auto& snapshot : snapshots_) {
        for (size_t i = 0; i < num_cameras; ++i) {
            const auto& image = reconstruction.Image(snapshot[i]);
            proj_centers_in_rig[i] =
                Inverse(CamFromRig(image.CameraId())).translation;
            proj_centers_in_world[i] = image.ProjectionCenter();
        }

        // Accumulate the relative scale for all pairs of camera distances.
        for (size_t i = 0; i < num_cameras; ++i) {
            for (size_t j = 0; j < i; ++j) {
                const double rig_dist =
                    (proj_centers_in_rig[i] - proj_centers_in_rig[j]).norm();
                const double world_dist =
                    (proj_centers_in_world[i] - proj_centers_in_world[j]).norm();
                const double kMinDist = 1e-6;
                if (rig_dist > kMinDist && world_dist > kMinDist) {
                    rig_from_world_scale += rig_dist / world_dist;
                    num_dists += 1;
                }
            }
        }
    }

    if (num_dists == 0) {
        return std::numeric_limits<double>::quiet_NaN();
    }

    return rig_from_world_scale / num_dists;
}

bool CameraRig::ComputeCamsFromRigs(const Reconstruction& reconstruction) {
    if (NumSnapshots() < 0)
    {
        LOG_ERR_OUT << "NumSnapshots()< 0";
        return false;
    }
    if (ref_camera_id_== kInvalidCameraId)
    {
        LOG_ERR_OUT << "ref_camera_id_== kInvalidCameraId";
        return false;
    }
    for (auto& cam_from_rig : cams_from_rigs_) {
        cam_from_rig.second.translation = Eigen::Vector3d::Zero();
    }

    std::unordered_map<camera_t, std::vector<Eigen::Quaterniond>>
        cam_from_ref_cam_rotations;
    for (const auto& snapshot : snapshots_) {
        // Find the image of the reference camera in the current snapshot.
        const Image* ref_image = nullptr;
        for (const auto image_id : snapshot) {
            const auto& image = reconstruction.Image(image_id);
            if (image.CameraId() == ref_camera_id_) {
                ref_image = &image;
                break;
            }
        }
        if (ref_image==nullptr)
        {
            LOG_ERR_OUT << "ref_image==nullptr";
            return false;
        }
        const Rigid3d world_from_ref_cam = Inverse(ref_image->CamFromWorld());

        // Compute the relative poses from all cameras in the current snapshot to
        // the reference camera.
        for (const auto image_id : snapshot) {
            const auto& image = reconstruction.Image(image_id);
            if (image.CameraId() != ref_camera_id_) {
                const Rigid3d cam_from_ref_cam =
                    image.CamFromWorld() * world_from_ref_cam;
                cam_from_ref_cam_rotations[image.CameraId()].push_back(
                    cam_from_ref_cam.rotation);
                CamFromRig(image.CameraId()).translation +=
                    cam_from_ref_cam.translation;
            }
        }
    }

    cams_from_rigs_.at(ref_camera_id_) = Rigid3d();

    // Compute the average relative poses.
    for (auto& cam_from_rig : cams_from_rigs_) {
        if (cam_from_rig.first != ref_camera_id_) {
            if (cam_from_ref_cam_rotations.count(cam_from_rig.first) == 0) {
                LOG_OUT<< "Need at least one snapshot with an image of camera "
                    << cam_from_rig.first << " and the reference camera "
                    << ref_camera_id_
                    << " to compute its relative pose in the camera rig";
                return false;
            }
            const std::vector<Eigen::Quaterniond>& cam_from_rig_rotations =
                cam_from_ref_cam_rotations.at(cam_from_rig.first);
            const std::vector<double> weights(cam_from_rig_rotations.size(), 1.0);
            cam_from_rig.second.rotation =
                AverageQuaternions(cam_from_rig_rotations, weights);
            cam_from_rig.second.translation /= cam_from_rig_rotations.size();
        }
    }
    return true;
}

Rigid3d CameraRig::ComputeRigFromWorld(
    const size_t snapshot_idx) const {
    return Rigid3d();
}
