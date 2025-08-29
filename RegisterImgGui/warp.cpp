#include "warp.h"
#include "eigen_alignment.h"
#include "log.h"
#include <Eigen/Geometry>


namespace {

    float GetPixelConstantBorder(const float* data,
        const int rows,
        const int cols,
        const int row,
        const int col) {
        if (row >= 0 && col >= 0 && row < rows && col < cols) {
            return data[row * cols + col];
        }
        else {
            return 0;
        }
    }

}  // namespace

void WarpImageBetweenCameras(const Camera& source_camera,
    const Camera& target_camera,
    const Bitmap& source_image,
    Bitmap* target_image) {
    THROW_CHECK_EQ(source_camera.width, source_image.Width());
    THROW_CHECK_EQ(source_camera.height, source_image.Height());
    THROW_CHECK_NOTNULL(target_image);

    target_image->Allocate(static_cast<int>(source_camera.width),
        static_cast<int>(source_camera.height),
        source_image.IsRGB());

    // To avoid aliasing, perform the warping in the source resolution and
    // then rescale the image at the end.
    Camera scaled_target_camera = target_camera;
    if (target_camera.width != source_camera.width ||
        target_camera.height != source_camera.height) {
        scaled_target_camera.Rescale(source_camera.width, source_camera.height);
    }

    Eigen::Vector2d image_point;
    for (int y = 0; y < target_image->Height(); ++y) {
        image_point.y() = y + 0.5;
        for (int x = 0; x < target_image->Width(); ++x) {
            image_point.x() = x + 0.5;

            // Camera models assume that the upper left pixel center is (0.5, 0.5).
            const Eigen::Vector2d cam_point =
                scaled_target_camera.CamFromImg(image_point);
            const Eigen::Vector2d source_point = source_camera.ImgFromCam(cam_point);

            BitmapColor<float> color;
            if (source_image.InterpolateBilinear(
                source_point.x() - 0.5, source_point.y() - 0.5, &color)) {
                target_image->SetPixel(x, y, color.Cast<uint8_t>());
            }
            else {
                target_image->SetPixel(x, y, BitmapColor<uint8_t>(0));
            }
        }
    }

    if (target_camera.width != source_camera.width ||
        target_camera.height != source_camera.height) {
        target_image->Rescale(target_camera.width, target_camera.height);
    }
}

void WarpImageBetweenCameras(const Camera& source_camera,
    const Camera& target_camera,
    const Bitmap& source_image,
    Bitmap* target_image,
    Eigen::MatrixXi& source_to_target_x_map,
    Eigen::MatrixXi& source_to_target_y_map
    ) {
    THROW_CHECK_EQ(source_camera.width, source_image.Width());
    THROW_CHECK_EQ(source_camera.height, source_image.Height());
    THROW_CHECK_NOTNULL(target_image);

    target_image->Allocate(static_cast<int>(source_camera.width),
        static_cast<int>(source_camera.height),
        source_image.IsRGB());

    // To avoid aliasing, perform the warping in the source resolution and
    // then rescale the image at the end.
    Camera scaled_target_camera = target_camera;
    if (target_camera.width != source_camera.width ||
        target_camera.height != source_camera.height) {
        scaled_target_camera.Rescale(source_camera.width, source_camera.height);
    }
    source_to_target_x_map = Eigen::MatrixXi::Zero(source_camera.height, source_camera.width);
    source_to_target_y_map = Eigen::MatrixXi::Zero(source_camera.height, source_camera.width);

    Eigen::Vector2d image_point;
    for (int y = 0; y < target_image->Height(); ++y) {
        image_point.y() = y + 0.5;
        for (int x = 0; x < target_image->Width(); ++x) {
            image_point.x() = x + 0.5;

            // Camera models assume that the upper left pixel center is (0.5, 0.5).
            const Eigen::Vector2d cam_point =
                scaled_target_camera.CamFromImg(image_point);
            const Eigen::Vector2d source_point = source_camera.ImgFromCam(cam_point);

            BitmapColor<float> color;
            if (source_image.InterpolateBilinear(
                source_point.x() - 0.5, source_point.y() - 0.5, &color)) {
                target_image->SetPixel(x, y, color.Cast<uint8_t>());

                int srcX = source_point.x() - 0.5;
                int srcY = source_point.y() - 0.5;
                if (srcX >= 0 && srcX < source_camera.width&& srcY >= 0 && srcY < source_camera.height)
                {
                    source_to_target_x_map(srcY, srcX) = x;
                    source_to_target_y_map(srcY, srcX) = y;
                }
            }
            else {
                target_image->SetPixel(x, y, BitmapColor<uint8_t>(0));
            }
        }
    }

    if (target_camera.width != source_camera.width ||
        target_camera.height != source_camera.height) {
        target_image->Rescale(target_camera.width, target_camera.height);
    }
}

void WarpImageWithHomography(const Eigen::Matrix3d& H,
    const Bitmap& source_image,
    Bitmap* target_image) {
    THROW_CHECK_NOTNULL(target_image);
    THROW_CHECK_GT(target_image->Width(), 0);
    THROW_CHECK_GT(target_image->Height(), 0);
    THROW_CHECK_EQ(source_image.IsRGB(), target_image->IsRGB());

    Eigen::Vector3d target_pixel(0, 0, 1);
    for (int y = 0; y < target_image->Height(); ++y) {
        target_pixel.y() = y + 0.5;
        for (int x = 0; x < target_image->Width(); ++x) {
            target_pixel.x() = x + 0.5;

            const Eigen::Vector2d source_pixel = (H * target_pixel).hnormalized();

            BitmapColor<float> color;
            if (source_image.InterpolateBilinear(
                source_pixel.x() - 0.5, source_pixel.y() - 0.5, &color)) {
                target_image->SetPixel(x, y, color.Cast<uint8_t>());
            }
            else {
                target_image->SetPixel(x, y, BitmapColor<uint8_t>(0));
            }
        }
    }
}

void WarpImageWithHomographyBetweenCameras(const Eigen::Matrix3d& H,
    const Camera& source_camera,
    const Camera& target_camera,
    const Bitmap& source_image,
    Bitmap* target_image) {
    THROW_CHECK_EQ(source_camera.width, source_image.Width());
    THROW_CHECK_EQ(source_camera.height, source_image.Height());
    THROW_CHECK_NOTNULL(target_image);

    target_image->Allocate(static_cast<int>(source_camera.width),
        static_cast<int>(source_camera.height),
        source_image.IsRGB());

    // To avoid aliasing, perform the warping in the source resolution and
    // then rescale the image at the end.
    Camera scaled_target_camera = target_camera;
    if (target_camera.width != source_camera.width ||
        target_camera.height != source_camera.height) {
        scaled_target_camera.Rescale(source_camera.width, source_camera.height);
    }

    Eigen::Vector3d image_point(0, 0, 1);
    for (int y = 0; y < target_image->Height(); ++y) {
        image_point.y() = y + 0.5;
        for (int x = 0; x < target_image->Width(); ++x) {
            image_point.x() = x + 0.5;

            // Camera models assume that the upper left pixel center is (0.5, 0.5).
            const Eigen::Vector3d warped_point = H * image_point;
            const Eigen::Vector2d cam_point =
                target_camera.CamFromImg(warped_point.hnormalized());
            const Eigen::Vector2d source_point = source_camera.ImgFromCam(cam_point);

            BitmapColor<float> color;
            if (source_image.InterpolateBilinear(
                source_point.x() - 0.5, source_point.y() - 0.5, &color)) {
                target_image->SetPixel(x, y, color.Cast<uint8_t>());
            }
            else {
                target_image->SetPixel(x, y, BitmapColor<uint8_t>(0));
            }
        }
    }

    if (target_camera.width != source_camera.width ||
        target_camera.height != source_camera.height) {
        target_image->Rescale(target_camera.width, target_camera.height);
    }
}

void ResampleImageBilinear(const float* data,
    const int rows,
    const int cols,
    const int new_rows,
    const int new_cols,
    float* resampled) {
    THROW_CHECK_NOTNULL(data);
    THROW_CHECK_NOTNULL(resampled);
    THROW_CHECK_GT(rows, 0);
    THROW_CHECK_GT(cols, 0);
    THROW_CHECK_GT(new_rows, 0);
    THROW_CHECK_GT(new_cols, 0);

    const float scale_r = static_cast<float>(rows) / static_cast<float>(new_rows);
    const float scale_c = static_cast<float>(cols) / static_cast<float>(new_cols);

    for (int r = 0; r < new_rows; ++r) {
        const float r_i = (r + 0.5f) * scale_r - 0.5f;
        const int r_i_min = std::floor(r_i);
        const int r_i_max = r_i_min + 1;
        const float d_r_min = r_i - r_i_min;
        const float d_r_max = r_i_max - r_i;

        for (int c = 0; c < new_cols; ++c) {
            const float c_i = (c + 0.5f) * scale_c - 0.5f;
            const int c_i_min = std::floor(c_i);
            const int c_i_max = c_i_min + 1;
            const float d_c_min = c_i - c_i_min;
            const float d_c_max = c_i_max - c_i;

            // Interpolation in column direction.
            const float value1 =
                d_c_max * GetPixelConstantBorder(data, rows, cols, r_i_min, c_i_min) +
                d_c_min * GetPixelConstantBorder(data, rows, cols, r_i_min, c_i_max);
            const float value2 =
                d_c_max * GetPixelConstantBorder(data, rows, cols, r_i_max, c_i_min) +
                d_c_min * GetPixelConstantBorder(data, rows, cols, r_i_max, c_i_max);

            // Interpolation in row direction.
            resampled[r * new_cols + c] = d_r_max * value1 + d_r_min * value2;
        }
    }
}

