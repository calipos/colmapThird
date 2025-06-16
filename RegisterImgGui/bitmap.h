#ifndef _BITMAP_H_
#define _BITMAP_H_

#include "colString.h"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <ios>
#include <limits>
#include <string>
#include <vector>
struct FIBITMAP;
// Templated bitmap color class.
template <typename T>
struct BitmapColor {
    BitmapColor();
    explicit BitmapColor(T gray);
    BitmapColor(T r, T g, T b);

    template <typename D>
    BitmapColor<D> Cast() const;

    bool operator==(const BitmapColor<T>& rhs) const;
    bool operator!=(const BitmapColor<T>& rhs) const;

    template <typename D>
    friend std::ostream& operator<<(std::ostream& output,
        const BitmapColor<D>& color);

    T r;
    T g;
    T b;
};

// Wrapper class around FreeImage bitmaps.
class Bitmap {
public:
    Bitmap();

    // Copy constructor.
    Bitmap(const Bitmap& other);
    // Move constructor.
    Bitmap(Bitmap&& other) noexcept;

    // Create bitmap object from existing FreeImage bitmap object. Note that
    // this class takes ownership of the object.
    explicit Bitmap(FIBITMAP* data);

    // Copy assignment.
    Bitmap& operator=(const Bitmap& other);
    // Move assignment.
    Bitmap& operator=(Bitmap&& other) noexcept;

    // Allocate bitmap by overwriting the existing data.
    bool Allocate(int width, int height, bool as_rgb);

    // Deallocate the bitmap by releasing the existing data.
    void Deallocate();

    // Get pointer to underlying FreeImage object.
    inline const FIBITMAP* Data() const;
    inline FIBITMAP* Data();

    // Dimensions of bitmap.
    inline int Width() const;
    inline int Height() const;
    inline int Channels() const;

    // Number of bits per pixel. This is 8 for grey and 24 for RGB image.
    unsigned int BitsPerPixel() const;

    // Scan width of bitmap which differs from the actual image width to achieve
    // 32 bit aligned memory. Also known as stride.
    unsigned int Pitch() const;

    // Check whether image is grey- or colorscale.
    inline bool IsRGB() const;
    inline bool IsGrey() const;

    // Number of bytes required to store image.
    size_t NumBytes() const;

    // Copy raw image data to array.
    std::vector<uint8_t> ConvertToRowMajorArray() const;
    std::vector<uint8_t> ConvertToColMajorArray() const;

    // Convert to/from raw bits.
    std::vector<uint8_t> ConvertToRawBits() const;
    static Bitmap ConvertFromRawBits(
        const uint8_t* data, int pitch, int width, int height, bool rgb = true);

    // Manipulate individual pixels. For grayscale images, only the red element
    // of the RGB color is used.
    bool GetPixel(int x, int y, BitmapColor<uint8_t>* color) const;
    bool SetPixel(int x, int y, const BitmapColor<uint8_t>& color);

    // Get pointer to y-th scanline, where the 0-th scanline is at the top.
    const uint8_t* GetScanline(int y) const;

    // Fill entire bitmap with uniform color. For grayscale images, the first
    // element of the vector is used.
    void Fill(const BitmapColor<uint8_t>& color);

    // Interpolate color at given floating point position.
    bool InterpolateNearestNeighbor(double x,
        double y,
        BitmapColor<uint8_t>* color) const;
    bool InterpolateBilinear(double x, double y, BitmapColor<float>* color) const;

    // Extract EXIF information from bitmap. Returns false if no EXIF information
    // is embedded in the bitmap.
    bool ExifCameraModel(std::string* camera_model) const;

    // Read bitmap at given path and convert to grey- or colorscale.
    bool Read(const std::string& path, bool as_rgb = true);

    // Write image to file. Flags can be used to set e.g. the JPEG quality.
    // Consult the FreeImage documentation for all available flags.
    bool Write(const std::string& path, int flags = 0) const;

    // Rescale image to the new dimensions.
    enum class RescaleFilter {
        kBilinear,
        kBox,
    };
    void Rescale(int new_width,
        int new_height,
        RescaleFilter filter = RescaleFilter::kBilinear);

    // Clone the image to a new bitmap object.
    Bitmap Clone() const;
    Bitmap CloneAsGrey() const;
    Bitmap CloneAsRGB() const;

    // Clone metadata from this bitmap object to another target bitmap object.
    void CloneMetadata(Bitmap* target) const;

private:
    struct FreeImageHandle {
        FreeImageHandle();
        explicit FreeImageHandle(FIBITMAP* ptr);
        ~FreeImageHandle();
        FreeImageHandle(FreeImageHandle&&) noexcept;
        FreeImageHandle& operator=(FreeImageHandle&&) noexcept;
        FreeImageHandle(const FreeImageHandle&) = delete;
        FreeImageHandle& operator=(const FreeImageHandle&) = delete;
        FIBITMAP* ptr;
    };

    void SetPtr(FIBITMAP* ptr);

    FreeImageHandle handle_;
    int width_;
    int height_;
    int channels_;
};

// Jet colormap inspired by Matlab. Grayvalues are expected in the range [0, 1]
// and are converted to RGB values in the same range.
class JetColormap {
public:
    static float Red(float gray);
    static float Green(float gray);
    static float Blue(float gray);

private:
    static float Interpolate(float val, float y0, float x0, float y1, float x1);
    static float Base(float val);
};

namespace internal {

    template <typename T1, typename T2>
    T2 BitmapColorCast(const T1 value) {
        return std::min(static_cast<T1>(std::numeric_limits<T2>::max()),
            std::max(static_cast<T1>(std::numeric_limits<T2>::min()),
                std::round(value)));
    }

}  // namespace internal

template <typename T>
BitmapColor<T>::BitmapColor() : r(0), g(0), b(0) {}

template <typename T>
BitmapColor<T>::BitmapColor(const T gray) : r(gray), g(gray), b(gray) {}

template <typename T>
BitmapColor<T>::BitmapColor(const T r, const T g, const T b)
    : r(r), g(g), b(b) {}

template <typename T>
template <typename D>
BitmapColor<D> BitmapColor<T>::Cast() const {
    BitmapColor<D> color;
    color.r = internal::BitmapColorCast<T, D>(r);
    color.g = internal::BitmapColorCast<T, D>(g);
    color.b = internal::BitmapColorCast<T, D>(b);
    return color;
}

template <typename T>
bool BitmapColor<T>::operator==(const BitmapColor<T>& rhs) const {
    return r == rhs.r && g == rhs.g && b == rhs.b;
}

template <typename T>
bool BitmapColor<T>::operator!=(const BitmapColor<T>& rhs) const {
    return r != rhs.r || g != rhs.g || b != rhs.b;
}

template <typename T>
std::ostream& operator<<(std::ostream& output, const BitmapColor<T>& color) {
    output << StringPrintf("RGB(%f, %f, %f)",
        static_cast<double>(color.r),
        static_cast<double>(color.g),
        static_cast<double>(color.b));
    return output;
}

FIBITMAP* Bitmap::Data() { return handle_.ptr; }
const FIBITMAP* Bitmap::Data() const { return handle_.ptr; }

int Bitmap::Width() const { return width_; }
int Bitmap::Height() const { return height_; }
int Bitmap::Channels() const { return channels_; }

bool Bitmap::IsRGB() const { return channels_ == 3; }

bool Bitmap::IsGrey() const { return channels_ == 1; }

#endif // !_BITMAP_H_
