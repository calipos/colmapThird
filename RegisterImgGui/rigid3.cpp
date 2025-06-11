#include "log.h"
#include "rigid3.h"

std::ostream& operator<<(std::ostream& stream, const Rigid3d& tform) {
    const static Eigen::IOFormat kVecFmt(
        Eigen::StreamPrecision, Eigen::DontAlignCols, ", ", ", ");
    stream << "Rigid3d(rotation_xyzw=[" << tform.rotation.coeffs().format(kVecFmt)
        << "], translation=[" << tform.translation.format(kVecFmt) << "])";
    return stream;
}