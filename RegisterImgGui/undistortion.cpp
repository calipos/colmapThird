#include "undistortion.h"
#include "colmath.h"
Camera UndistortCamera(const UndistortCameraOptions& options,
    const Camera& camera) {
    //THROW_CHECK_GE(options.blank_pixels, 0);
    //THROW_CHECK_LE(options.blank_pixels, 1);
    //THROW_CHECK_GT(options.min_scale, 0.0);
    //THROW_CHECK_LE(options.min_scale, options.max_scale);
    //THROW_CHECK_NE(options.max_image_size, 0);
    //THROW_CHECK_GE(options.roi_min_x, 0.0);
    //THROW_CHECK_GE(options.roi_min_y, 0.0);
    //THROW_CHECK_LE(options.roi_max_x, 1.0);
    //THROW_CHECK_LE(options.roi_max_y, 1.0);
    //THROW_CHECK_LT(options.roi_min_x, options.roi_max_x);
    //THROW_CHECK_LT(options.roi_min_y, options.roi_max_y);

    Camera undistorted_camera;
    undistorted_camera.model_id = PinholeCameraModel::model_id;
    undistorted_camera.width = camera.width;
    undistorted_camera.height = camera.height;
    undistorted_camera.params.resize(PinholeCameraModel::num_params, 0);

    // Copy focal length parameters.
    const span<const size_t> focal_length_idxs = camera.FocalLengthIdxs();
    //THROW_CHECK_LE(focal_length_idxs.size(), 2) << "Not more than two focal length parameters supported.";
    undistorted_camera.SetFocalLengthX(camera.FocalLengthX());
    undistorted_camera.SetFocalLengthY(camera.FocalLengthY());

    // Copy principal point parameters.
    undistorted_camera.SetPrincipalPointX(camera.PrincipalPointX());
    undistorted_camera.SetPrincipalPointY(camera.PrincipalPointY());

    // Modify undistorted camera parameters based on ROI if enabled
    size_t roi_min_x = 0;
    size_t roi_min_y = 0;
    size_t roi_max_x = camera.width;
    size_t roi_max_y = camera.height;

    const bool roi_enabled = options.roi_min_x > 0.0 || options.roi_min_y > 0.0 ||
        options.roi_max_x < 1.0 || options.roi_max_y < 1.0;

    if (roi_enabled) {
        roi_min_x = static_cast<size_t>(
            std::round(options.roi_min_x * static_cast<double>(camera.width)));
        roi_min_y = static_cast<size_t>(
            std::round(options.roi_min_y * static_cast<double>(camera.height)));
        roi_max_x = static_cast<size_t>(
            std::round(options.roi_max_x * static_cast<double>(camera.width)));
        roi_max_y = static_cast<size_t>(
            std::round(options.roi_max_y * static_cast<double>(camera.height)));

        // Make sure that the roi is valid.
        roi_min_x = std::min(roi_min_x, camera.width - 1);
        roi_min_y = std::min(roi_min_y, camera.height - 1);
        roi_max_x = std::max(roi_max_x, roi_min_x + 1);
        roi_max_y = std::max(roi_max_y, roi_min_y + 1);

        undistorted_camera.width = roi_max_x - roi_min_x;
        undistorted_camera.height = roi_max_y - roi_min_y;

        undistorted_camera.SetPrincipalPointX(camera.PrincipalPointX() -
            static_cast<double>(roi_min_x));
        undistorted_camera.SetPrincipalPointY(camera.PrincipalPointY() -
            static_cast<double>(roi_min_y));
    }

    // Scale the image such the the boundary of the undistorted image.
    if (roi_enabled || (camera.model_id != SimplePinholeCameraModel::model_id &&
        camera.model_id != PinholeCameraModel::model_id)) {
        // Determine min/max coordinates along top / bottom image border.

        double left_min_x = std::numeric_limits<double>::max();
        double left_max_x = std::numeric_limits<double>::lowest();
        double right_min_x = std::numeric_limits<double>::max();
        double right_max_x = std::numeric_limits<double>::lowest();

        for (size_t y = roi_min_y; y < roi_max_y; ++y) {
            // Left border.
            const Eigen::Vector2d point1_in_cam =
                camera.CamFromImg(Eigen::Vector2d(0.5, y + 0.5));
            const Eigen::Vector2d undistorted_point1 =
                undistorted_camera.ImgFromCam(point1_in_cam);
            left_min_x = std::min(left_min_x, undistorted_point1(0));
            left_max_x = std::max(left_max_x, undistorted_point1(0));
            // Right border.
            const Eigen::Vector2d point2_in_cam =
                camera.CamFromImg(Eigen::Vector2d(camera.width - 0.5, y + 0.5));
            const Eigen::Vector2d undistorted_point2 =
                undistorted_camera.ImgFromCam(point2_in_cam);
            right_min_x = std::min(right_min_x, undistorted_point2(0));
            right_max_x = std::max(right_max_x, undistorted_point2(0));
        }

        // Determine min, max coordinates along left / right image border.

        double top_min_y = std::numeric_limits<double>::max();
        double top_max_y = std::numeric_limits<double>::lowest();
        double bottom_min_y = std::numeric_limits<double>::max();
        double bottom_max_y = std::numeric_limits<double>::lowest();

        for (size_t x = roi_min_x; x < roi_max_x; ++x) {
            // Top border.
            const Eigen::Vector2d point1_in_cam =
                camera.CamFromImg(Eigen::Vector2d(x + 0.5, 0.5));
            const Eigen::Vector2d undistorted_point1 =
                undistorted_camera.ImgFromCam(point1_in_cam);
            top_min_y = std::min(top_min_y, undistorted_point1(1));
            top_max_y = std::max(top_max_y, undistorted_point1(1));
            // Bottom border.
            const Eigen::Vector2d point2_in_cam =
                camera.CamFromImg(Eigen::Vector2d(x + 0.5, camera.height - 0.5));
            const Eigen::Vector2d undistorted_point2 =
                undistorted_camera.ImgFromCam(point2_in_cam);
            bottom_min_y = std::min(bottom_min_y, undistorted_point2(1));
            bottom_max_y = std::max(bottom_max_y, undistorted_point2(1));
        }

        const double cx = undistorted_camera.PrincipalPointX();
        const double cy = undistorted_camera.PrincipalPointY();

        // Scale such that undistorted image contains all pixels of distorted image.
        const double min_scale_x =
            std::min(cx / (cx - left_min_x),
                (undistorted_camera.width - 0.5 - cx) / (right_max_x - cx));
        const double min_scale_y =
            std::min(cy / (cy - top_min_y),
                (undistorted_camera.height - 0.5 - cy) / (bottom_max_y - cy));

        // Scale such that there are no blank pixels in undistorted image.
        const double max_scale_x =
            std::max(cx / (cx - left_max_x),
                (undistorted_camera.width - 0.5 - cx) / (right_min_x - cx));
        const double max_scale_y =
            std::max(cy / (cy - top_max_y),
                (undistorted_camera.height - 0.5 - cy) / (bottom_min_y - cy));

        // Interpolate scale according to blank_pixels.
        double scale_x = 1.0 / (min_scale_x * options.blank_pixels +
            max_scale_x * (1.0 - options.blank_pixels));
        double scale_y = 1.0 / (min_scale_y * options.blank_pixels +
            max_scale_y * (1.0 - options.blank_pixels));

        // Clip the scaling factors.
        scale_x = Clamp(scale_x, options.min_scale, options.max_scale);
        scale_y = Clamp(scale_y, options.min_scale, options.max_scale);

        // Scale undistorted camera dimensions.
        const size_t orig_undistorted_camera_width = undistorted_camera.width;
        const size_t orig_undistorted_camera_height = undistorted_camera.height;
        undistorted_camera.width =
            static_cast<size_t>(std::max(1.0, scale_x * undistorted_camera.width));
        undistorted_camera.height =
            static_cast<size_t>(std::max(1.0, scale_y * undistorted_camera.height));

        // Scale the principal point according to the new dimensions of the camera.
        undistorted_camera.SetPrincipalPointX(
            undistorted_camera.PrincipalPointX() *
            static_cast<double>(undistorted_camera.width) /
            static_cast<double>(orig_undistorted_camera_width));
        undistorted_camera.SetPrincipalPointY(
            undistorted_camera.PrincipalPointY() *
            static_cast<double>(undistorted_camera.height) /
            static_cast<double>(orig_undistorted_camera_height));
    }

    if (options.max_image_size > 0) {
        const double max_image_scale_x =
            options.max_image_size / static_cast<double>(undistorted_camera.width);
        const double max_image_scale_y =
            options.max_image_size / static_cast<double>(undistorted_camera.height);
        const double max_image_scale =
            std::min(max_image_scale_x, max_image_scale_y);
        if (max_image_scale < 1.0) {
            undistorted_camera.Rescale(max_image_scale);
        }
    }

    return undistorted_camera;
}

void UndistortImage(const UndistortCameraOptions& options,
    const Bitmap& distorted_bitmap,
    const Camera& distorted_camera,
    Bitmap* undistorted_bitmap,
    Camera* undistorted_camera) {
    //THROW_CHECK_EQ(distorted_camera.width, distorted_bitmap.Width());
    //THROW_CHECK_EQ(distorted_camera.height, distorted_bitmap.Height());

    *undistorted_camera = UndistortCamera(options, distorted_camera);

    //WarpImageBetweenCameras(distorted_camera,
    //    *undistorted_camera,
    //    distorted_bitmap,
    //    undistorted_bitmap);

    distorted_bitmap.CloneMetadata(undistorted_bitmap);
}

