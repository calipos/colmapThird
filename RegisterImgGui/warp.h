#ifndef _WARP_H_
#define _WARP_H_

#include "camera.h"
#include "bitmap.h"

// Warp source image to target image by projecting the pixels of the target
// image up to infinity and projecting it down into the source image
// (i.e. an inverse mapping). The function allocates the target image.
void WarpImageBetweenCameras(const Camera& source_camera,
    const Camera& target_camera,
    const Bitmap& source_image,
    Bitmap* target_image);

// Warp an image with the given homography, where H defines the pixel mapping
// from the target to source image. Note that the pixel centers are assumed to
// have coordinates (0.5, 0.5).
void WarpImageWithHomography(const Eigen::Matrix3d& H,
    const Bitmap& source_image,
    Bitmap* target_image);

// First, warp source image to target image by projecting the pixels of the
// target image up to infinity and projecting it down into the source image
// (i.e. an inverse mapping). Second, warp the coordinates from the first
// warping with the given homography. The function allocates the target image.
void WarpImageWithHomographyBetweenCameras(const Eigen::Matrix3d& H,
    const Camera& source_camera,
    const Camera& target_camera,
    const Bitmap& source_image,
    Bitmap* target_image);

// Resample row-major image using bilinear interpolation.
void ResampleImageBilinear(const float* data,
    int rows,
    int cols,
    int new_rows,
    int new_cols,
    float* resampled);

 

#endif // !_WARP_H_
