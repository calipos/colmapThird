#ifndef _UNDISTORTION_H_
#define _UNDISTORTION_H_
#include "bitmap.h"
#include "camera.h"
struct UndistortCameraOptions {
	// The amount of blank pixels in the undistorted image in the range [0, 1].
	double blank_pixels = 0.0;

	// Minimum and maximum scale change of camera used to satisfy the blank
	// pixel constraint.
	double min_scale = 0.2;
	double max_scale = 2.0;

	// Maximum image size in terms of width or height of the undistorted camera.
	int max_image_size = -1;

	// The 4 factors in the range [0, 1] that define the ROI (region of interest)
	// in original image. The bounding box pixel coordinates are calculated as
	//    (roi_min_x * Width, roi_min_y * Height) and
	//    (roi_max_x * Width, roi_max_y * Height).
	double roi_min_x = 0.0;
	double roi_min_y = 0.0;
	double roi_max_x = 1.0;
	double roi_max_y = 1.0;
};
Camera UndistortCamera(const UndistortCameraOptions& options, const Camera& camera);
// Undistort image such that the viewing geometry of the undistorted image
// follows a pinhole camera model. See `UndistortCamera` for more details
// on the undistortion conventions.
void UndistortImage(const UndistortCameraOptions& options,
	const Bitmap& distorted_image,
	const Camera& distorted_camera,
	Bitmap* undistorted_image,
	Camera* undistorted_camera);
#endif // !_UNDISTORTION_H_
