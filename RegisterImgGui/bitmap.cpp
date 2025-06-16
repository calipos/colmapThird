#include "bitmap.h"
#include "colmath.h"
#include "log.h"
#include "misc.h"
#include <regex>
#include <unordered_map>
#include <filesystem>
#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <Windows.h>
#endif
#include <FreeImage.h>
#ifdef FREEIMAGE_LIB  // Only needed for static FreeImage.
struct FreeImageInitializer {
	FreeImageInitializer() { FreeImage_Initialise(); }
	~FreeImageInitializer() { FreeImage_DeInitialise(); }
};
const static auto initializer = FreeImageInitializer();
#endif  // FREEIMAGE_LIB
bool ReadExifTag(FIBITMAP* ptr,
    const FREE_IMAGE_MDMODEL model,
    const std::string& tag_name,
    std::string* result) {
    FITAG* tag = nullptr;
    FreeImage_GetMetadata(model, ptr, tag_name.c_str(), &tag);
    if (tag == nullptr) {
        *result = "";
        return false;
    }
    else {
        if (tag_name == "FocalPlaneXResolution") {
            // This tag seems to be in the wrong category.
            *result = std::string(FreeImage_TagToString(FIMD_EXIF_INTEROP, tag));
        }
        else {
            *result = FreeImage_TagToString(model, tag);
        }
        return true;
    }
}

bool IsPtrGrey(FIBITMAP* ptr) {
    return FreeImage_GetColorType(ptr) == FIC_MINISBLACK &&
        FreeImage_GetBPP(ptr) == 8;
}

bool IsPtrRGB(FIBITMAP* ptr) {
    return FreeImage_GetColorType(ptr) == FIC_RGB && FreeImage_GetBPP(ptr) == 24;
}

bool IsPtrSupported(FIBITMAP* ptr) { return IsPtrGrey(ptr) || IsPtrRGB(ptr); }


Bitmap::Bitmap() : width_(0), height_(0), channels_(0) {}

Bitmap::Bitmap(const Bitmap& other) : Bitmap() {
    if (other.handle_.ptr != nullptr) {
        SetPtr(FreeImage_Clone(other.handle_.ptr));
    }
}

Bitmap::Bitmap(Bitmap&& other) noexcept : Bitmap() {
    handle_ = std::move(other.handle_);
    width_ = other.width_;
    height_ = other.height_;
    channels_ = other.channels_;
    other.width_ = 0;
    other.height_ = 0;
    other.channels_ = 0;
}

Bitmap::Bitmap(FIBITMAP* data) : Bitmap() { SetPtr(data); }

Bitmap& Bitmap::operator=(const Bitmap& other) {
    if (other.handle_.ptr != nullptr) {
        SetPtr(FreeImage_Clone(other.handle_.ptr));
    }
    return *this;
}

Bitmap& Bitmap::operator=(Bitmap&& other) noexcept {
    if (this != &other) {
        handle_ = std::move(other.handle_);
        width_ = other.width_;
        height_ = other.height_;
        channels_ = other.channels_;
        other.width_ = 0;
        other.height_ = 0;
        other.channels_ = 0;
    }
    return *this;
}

bool Bitmap::Allocate(const int width, const int height, const bool as_rgb) {
    width_ = width;
    height_ = height;
    if (as_rgb) {
        const int kNumBitsPerPixel = 24;
        handle_ =
            FreeImageHandle(FreeImage_Allocate(width, height, kNumBitsPerPixel));
        channels_ = 3;
    }
    else {
        const int kNumBitsPerPixel = 8;
        handle_ =
            FreeImageHandle(FreeImage_Allocate(width, height, kNumBitsPerPixel));
        channels_ = 1;
    }
    return handle_.ptr != nullptr;
}

void Bitmap::Deallocate() {
    handle_ = FreeImageHandle();
    width_ = 0;
    height_ = 0;
    channels_ = 0;
}

size_t Bitmap::NumBytes() const {
    if (handle_.ptr != nullptr) {
        return Pitch() * height_;
    }
    else {
        return 0;
    }
}

unsigned int Bitmap::BitsPerPixel() const {
    return FreeImage_GetBPP(handle_.ptr);
}

unsigned int Bitmap::Pitch() const { return FreeImage_GetPitch(handle_.ptr); }

std::vector<uint8_t> Bitmap::ConvertToRowMajorArray() const {
    std::vector<uint8_t> array(width_ * height_ * channels_);
    size_t i = 0;
    for (int y = 0; y < height_; ++y) {
        const uint8_t* line = FreeImage_GetScanLine(handle_.ptr, height_ - 1 - y);
        for (int x = 0; x < width_; ++x) {
            for (int d = 0; d < channels_; ++d) {
                array[i] = line[x * channels_ + d];
                i += 1;
            }
        }
    }
    return array;
}

std::vector<uint8_t> Bitmap::ConvertToColMajorArray() const {
    std::vector<uint8_t> array(width_ * height_ * channels_);
    size_t i = 0;
    for (int d = 0; d < channels_; ++d) {
        for (int x = 0; x < width_; ++x) {
            for (int y = 0; y < height_; ++y) {
                const uint8_t* line =
                    FreeImage_GetScanLine(handle_.ptr, height_ - 1 - y);
                array[i] = line[x * channels_ + d];
                i += 1;
            }
        }
    }
    return array;
}

std::vector<uint8_t> Bitmap::ConvertToRawBits() const {
    const unsigned int pitch = Pitch();
    const unsigned int bpp = BitsPerPixel();
    std::vector<uint8_t> raw_bits(pitch * height_ * bpp / 8, 0);
    FreeImage_ConvertToRawBits(raw_bits.data(),
        handle_.ptr,
        pitch,
        bpp,
        FI_RGBA_RED_MASK,
        FI_RGBA_GREEN_MASK,
        FI_RGBA_BLUE_MASK,
        /*topdown=*/true);
    return raw_bits;
}

Bitmap Bitmap::ConvertFromRawBits(
    const uint8_t* data, int pitch, int width, int height, bool rgb) {
    const unsigned bpp = rgb ? 24 : 8;
    return Bitmap(FreeImage_ConvertFromRawBitsEx(/*copy_source=*/true,
        const_cast<uint8_t*>(data),
        FIT_BITMAP,
        width,
        height,
        pitch,
        bpp,
        FI_RGBA_RED_MASK,
        FI_RGBA_GREEN_MASK,
        FI_RGBA_BLUE_MASK,
        /*topdown=*/true));
}

bool Bitmap::GetPixel(const int x,
    const int y,
    BitmapColor<uint8_t>* color) const {
    if (x < 0 || x >= width_ || y < 0 || y >= height_) {
        return false;
    }

    const uint8_t* line = FreeImage_GetScanLine(handle_.ptr, height_ - 1 - y);

    if (IsGrey()) {
        color->r = line[x];
        return true;
    }
    else if (IsRGB()) {
        color->r = line[3 * x + FI_RGBA_RED];
        color->g = line[3 * x + FI_RGBA_GREEN];
        color->b = line[3 * x + FI_RGBA_BLUE];
        return true;
    }

    return false;
}

bool Bitmap::SetPixel(const int x,
    const int y,
    const BitmapColor<uint8_t>& color) {
    if (x < 0 || x >= width_ || y < 0 || y >= height_) {
        return false;
    }

    uint8_t* line = FreeImage_GetScanLine(handle_.ptr, height_ - 1 - y);

    if (IsGrey()) {
        line[x] = color.r;
        return true;
    }
    else if (IsRGB()) {
        line[3 * x + FI_RGBA_RED] = color.r;
        line[3 * x + FI_RGBA_GREEN] = color.g;
        line[3 * x + FI_RGBA_BLUE] = color.b;
        return true;
    }

    return false;
}

const uint8_t* Bitmap::GetScanline(const int y) const {
    if (y < 0)
    {
        LOG_ERR_OUT << "y<0";
        return nullptr;
    }
    if (y >= height_)
    {
        LOG_ERR_OUT << "y >= height_";
        return nullptr;
    }
    return FreeImage_GetScanLine(handle_.ptr, height_ - 1 - y);
}

void Bitmap::Fill(const BitmapColor<uint8_t>& color) {
    for (int y = 0; y < height_; ++y) {
        uint8_t* line = FreeImage_GetScanLine(handle_.ptr, height_ - 1 - y);
        for (int x = 0; x < width_; ++x) {
            if (IsGrey()) {
                line[x] = color.r;
            }
            else if (IsRGB()) {
                line[3 * x + FI_RGBA_RED] = color.r;
                line[3 * x + FI_RGBA_GREEN] = color.g;
                line[3 * x + FI_RGBA_BLUE] = color.b;
            }
        }
    }
}

bool Bitmap::InterpolateNearestNeighbor(const double x,
    const double y,
    BitmapColor<uint8_t>* color) const {
    const int xx = static_cast<int>(std::round(x));
    const int yy = static_cast<int>(std::round(y));
    return GetPixel(xx, yy, color);
}

bool Bitmap::InterpolateBilinear(const double x,
    const double y,
    BitmapColor<float>* color) const {
    // FreeImage's coordinate system origin is in the lower left of the image.
    const double inv_y = height_ - 1 - y;

    const int x0 = static_cast<int>(std::floor(x));
    const int x1 = x0 + 1;
    const int y0 = static_cast<int>(std::floor(inv_y));
    const int y1 = y0 + 1;

    if (x0 < 0 || x1 >= width_ || y0 < 0 || y1 >= height_) {
        return false;
    }

    const double dx = x - x0;
    const double dy = inv_y - y0;
    const double dx_1 = 1 - dx;
    const double dy_1 = 1 - dy;

    const uint8_t* line0 = FreeImage_GetScanLine(handle_.ptr, y0);
    const uint8_t* line1 = FreeImage_GetScanLine(handle_.ptr, y1);

    if (IsGrey()) {
        // Top row, column-wise linear interpolation.
        const double v0 = dx_1 * line0[x0] + dx * line0[x1];

        // Bottom row, column-wise linear interpolation.
        const double v1 = dx_1 * line1[x0] + dx * line1[x1];

        // Row-wise linear interpolation.
        color->r = dy_1 * v0 + dy * v1;
        return true;
    }
    else if (IsRGB()) {
        const uint8_t* p00 = &line0[3 * x0];
        const uint8_t* p01 = &line0[3 * x1];
        const uint8_t* p10 = &line1[3 * x0];
        const uint8_t* p11 = &line1[3 * x1];

        // Top row, column-wise linear interpolation.
        const double v0_r = dx_1 * p00[FI_RGBA_RED] + dx * p01[FI_RGBA_RED];
        const double v0_g = dx_1 * p00[FI_RGBA_GREEN] + dx * p01[FI_RGBA_GREEN];
        const double v0_b = dx_1 * p00[FI_RGBA_BLUE] + dx * p01[FI_RGBA_BLUE];

        // Bottom row, column-wise linear interpolation.
        const double v1_r = dx_1 * p10[FI_RGBA_RED] + dx * p11[FI_RGBA_RED];
        const double v1_g = dx_1 * p10[FI_RGBA_GREEN] + dx * p11[FI_RGBA_GREEN];
        const double v1_b = dx_1 * p10[FI_RGBA_BLUE] + dx * p11[FI_RGBA_BLUE];

        // Row-wise linear interpolation.
        color->r = dy_1 * v0_r + dy * v1_r;
        color->g = dy_1 * v0_g + dy * v1_g;
        color->b = dy_1 * v0_b + dy * v1_b;
        return true;
    }

    return false;
}

bool Bitmap::ExifCameraModel(std::string* camera_model) const {
    // Read camera make and model
    std::string make_str;
    std::string model_str;
    std::string focal_length;
    *camera_model = "";
    if (ReadExifTag(handle_.ptr, FIMD_EXIF_MAIN, "Make", &make_str)) {
        *camera_model += (make_str + "-");
    }
    else {
        *camera_model = "";
        return false;
    }
    if (ReadExifTag(handle_.ptr, FIMD_EXIF_MAIN, "Model", &model_str)) {
        *camera_model += (model_str + "-");
    }
    else {
        *camera_model = "";
        return false;
    }
    if (ReadExifTag(handle_.ptr,
        FIMD_EXIF_EXIF,
        "FocalLengthIn35mmFilm",
        &focal_length) ||
        ReadExifTag(handle_.ptr, FIMD_EXIF_EXIF, "FocalLength", &focal_length)) {
        *camera_model += (focal_length + "-");
    }
    else {
        *camera_model = "";
        return false;
    }
    *camera_model += (std::to_string(width_) + "x" + std::to_string(height_));
    return true;
}


bool Bitmap::Read(const std::string& path, const bool as_rgb) {
    if (!std::filesystem::exists(path)) {
        return false;
    }

    const FREE_IMAGE_FORMAT format = FreeImage_GetFileType(path.c_str(), 0);

    if (format == FIF_UNKNOWN) {
        return false;
    }

    handle_ = FreeImageHandle(FreeImage_Load(format, path.c_str()));
    if (handle_.ptr == nullptr) {
        return false;
    }

    if (!IsPtrRGB(handle_.ptr) && as_rgb) {
        FIBITMAP* converted_bitmap = FreeImage_ConvertTo24Bits(handle_.ptr);
        handle_ = FreeImageHandle(converted_bitmap);
    }
    else if (!IsPtrGrey(handle_.ptr) && !as_rgb) {
        if (FreeImage_GetBPP(handle_.ptr) != 24) {
            FIBITMAP* converted_bitmap_24 = FreeImage_ConvertTo24Bits(handle_.ptr);
            handle_ = FreeImageHandle(converted_bitmap_24);
        }
        FIBITMAP* converted_bitmap = FreeImage_ConvertToGreyscale(handle_.ptr);
        handle_ = FreeImageHandle(converted_bitmap);
    }

    if (!IsPtrSupported(handle_.ptr)) {
        handle_ = FreeImageHandle();
        return false;
    }

    width_ = FreeImage_GetWidth(handle_.ptr);
    height_ = FreeImage_GetHeight(handle_.ptr);
    channels_ = as_rgb ? 3 : 1;

    return true;
}

bool Bitmap::Write(const std::string& path, const int flags) const {
    FREE_IMAGE_FORMAT save_format = FreeImage_GetFIFFromFilename(path.c_str());
    if (save_format == FIF_UNKNOWN) {
        // If format could not be deduced, save as PNG by default.
        save_format = FIF_PNG;
    }

    int save_flags = flags;
    if (save_format == FIF_JPEG && flags == 0) {
        // Use superb JPEG quality by default to avoid artifacts.
        save_flags = JPEG_QUALITYSUPERB;
    }

    bool success = false;
    if (save_flags == 0) {
        success = FreeImage_Save(save_format, handle_.ptr, path.c_str());
    }
    else {
        success =
            FreeImage_Save(save_format, handle_.ptr, path.c_str(), save_flags);
    }

    return success;
}


void Bitmap::Rescale(const int new_width,
    const int new_height,
    RescaleFilter filter) {
    FREE_IMAGE_FILTER fi_filter = FILTER_BILINEAR;
    switch (filter) {
    case RescaleFilter::kBilinear:
        fi_filter = FILTER_BILINEAR;
        break;
    case RescaleFilter::kBox:
        fi_filter = FILTER_BOX;
        break;
    default:
        LOG_ERR_OUT << "Filter not implemented";
    }
    SetPtr(FreeImage_Rescale(handle_.ptr, new_width, new_height, fi_filter));
}

Bitmap Bitmap::Clone() const {
    FIBITMAP* cloned = FreeImage_Clone(handle_.ptr);
    return Bitmap(cloned);
}

Bitmap Bitmap::CloneAsGrey() const {
    if (IsGrey()) {
        return Clone();
    }
    else {
        return Bitmap(FreeImage_ConvertToGreyscale(handle_.ptr));
    }
}

Bitmap Bitmap::CloneAsRGB() const {
    if (IsRGB()) {
        return Clone();
    }
    else {
        return Bitmap(FreeImage_ConvertTo24Bits(handle_.ptr));
    }
}

void Bitmap::CloneMetadata(Bitmap* target) const {
    if (target==nullptr)
    {
        LOG_ERR_OUT << "nullptr";
        return;
    }
    if (target->Data() == nullptr)
    {
        LOG_ERR_OUT << "nullptr";
        return;
    }
    FreeImage_CloneMetadata(handle_.ptr, target->Data());
}

void Bitmap::SetPtr(FIBITMAP* ptr) {
    if (ptr == nullptr)
    {
        LOG_ERR_OUT << "nullptr";
        return;
    }
    if (!IsPtrSupported(ptr)) {
        FreeImageHandle temp_handle(ptr);
        ptr = FreeImage_ConvertTo24Bits(temp_handle.ptr);
        if (!IsPtrSupported(ptr))
        {
            LOG_ERR_OUT << "Not Supported";
            return;
        }
    }

    handle_ = FreeImageHandle(ptr);
    width_ = FreeImage_GetWidth(handle_.ptr);
    height_ = FreeImage_GetHeight(handle_.ptr);
    channels_ = IsPtrRGB(handle_.ptr) ? 3 : 1;
}

Bitmap::FreeImageHandle::FreeImageHandle() : ptr(nullptr) {}

Bitmap::FreeImageHandle::FreeImageHandle(FIBITMAP* ptr) : ptr(ptr) {}

Bitmap::FreeImageHandle::~FreeImageHandle() {
    if (ptr != nullptr) {
        FreeImage_Unload(ptr);
        ptr = nullptr;
    }
}

Bitmap::FreeImageHandle::FreeImageHandle(
    Bitmap::FreeImageHandle&& other) noexcept {
    ptr = other.ptr;
    other.ptr = nullptr;
}

Bitmap::FreeImageHandle& Bitmap::FreeImageHandle::operator=(
    Bitmap::FreeImageHandle&& other) noexcept {
    if (this != &other) {
        if (ptr != nullptr) {
            FreeImage_Unload(ptr);
        }
        ptr = other.ptr;
        other.ptr = nullptr;
    }
    return *this;
}

float JetColormap::Red(const float gray) { return Base(gray - 0.25f); }

float JetColormap::Green(const float gray) { return Base(gray); }

float JetColormap::Blue(const float gray) { return Base(gray + 0.25f); }

float JetColormap::Base(const float val) {
    // NOLINTNEXTLINE(bugprone-branch-clone)
    if (val <= 0.125f) {
        return 0.0f;
    }
    else if (val <= 0.375f) {
        return Interpolate(2.0f * val - 1.0f, 0.0f, -0.75f, 1.0f, -0.25f);
    }
    else if (val <= 0.625f) {
        return 1.0f;
    }
    else if (val <= 0.87f) {
        return Interpolate(2.0f * val - 1.0f, 1.0f, 0.25f, 0.0f, 0.75f);
    }
    else {
        return 0.0f;
    }
}

float JetColormap::Interpolate(const float val,
    const float y0,
    const float x0,
    const float y1,
    const float x1) {
    return (val - x0) * (y1 - y0) / (x1 - x0) + y0;
}