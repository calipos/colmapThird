#ifndef _DNN_HELPER_H_
#define _DNN_HELPER_H_
#include <fstream>
#include <optional>
#include <numeric>
#include "log.h"
#include "net.h"
#include "opencv2/opencv.hpp"

namespace dnn
{
    namespace ncnnHelper
    {
        cv::dnn::MatShape getBlobShape(const ncnn::Mat& out);
        void convertImgToMemFile(const std::string& path);
        cv::Mat recoverFromMemfile(const std::string& path);
        std::vector<int>getDenominators(const cv::dnn::MatShape& shape);
        std::vector<int>getPos(const int& idx, const std::vector<int>& denominators);
        std::ostream& printBlob(const ncnn::Mat& out, std::ostream& os=std::cout);
        bool dataHasNanInf(const ncnn::Mat& out);
        void writeBlob(const std::string& path, const ncnn::Mat& out);
        bool serializationBlob(const ncnn::Mat& out, cv::dnn::MatShape& shape, std::vector<float>& dat);
        std::ostream& operator<<(std::ostream& os, const ncnn::Mat& out);
    }

    namespace ocvHelper
    {
        enum class OnnxType
        {
            onnx_float32 = 1,
            onnx_uint8 = 2,
            onnx_int8 = 3,
            onnx_uint16 = 4,
            onnx_int16 = 5,
            onnx_int32 = 6,
            onnx_int64 = 7,
            onnx_string = 8,
            onnx_bool = 9,
            onnx_float16 = 10,
            onnx_float64 = 11,
            onnx_uint32 = 12,
            onnx_uint64 = 14,
        };
        std::ostream& operator<<(std::ostream& os, const cv::dnn::MatShape& shape);
        bool operator==(const cv::dnn::MatShape& shape1, const cv::dnn::MatShape& shape2);
        cv::dnn::MatShape getBlobShape(const cv::Mat& blob);
        template<typename dtype>
        class Blob
        {
        public:
            const OnnxType& getType()const
            {
                return type;
            }
            const dtype* getData()const
            {
                return (const dtype*)data;
            }
            dtype* getMutableData()
            {
                return data;
            }
            bool setOtherData(dtype* otherData)
            {
                if (needManualDestroy && data != nullptr)
                {
                    delete[]data;
                    data = nullptr;
                }
                data = otherData;
                return true;
            }
            const cv::dnn::MatShape getShape()const
            {
                return shape;
            }
            cv::dnn::MatShape& getMutableShape()
            {
                return shape;
            }
            Blob() = default;
            Blob(const cv::Mat& blob_)
            {
                data = (dtype*)blob_.data;
                shape = getBlobShape(blob_);
                if (std::is_same<dtype, float>::value)
                {
                    type = OnnxType::onnx_float32;
                }
                else if (std::is_same<dtype, std::uint8_t>::value)
                {
                    type = OnnxType::onnx_uint8;
                }
                else if (std::is_same<dtype, std::int8_t>::value)
                {
                    type = OnnxType::onnx_int8;
                }
                else if (std::is_same<dtype, std::uint16_t>::value)
                {
                    type = OnnxType::onnx_uint16;
                }
                else if (std::is_same<dtype, std::int16_t>::value)
                {
                    type = OnnxType::onnx_int16;
                }
                else if (std::is_same<dtype, int>::value)
                {
                    type = OnnxType::onnx_int32;
                }
                else if (std::is_same<dtype, std::int64_t>::value)
                {
                    type = OnnxType::onnx_int64;
                }
                else if (std::is_same<dtype, bool>::value)
                {
                    type = OnnxType::onnx_bool;
                }
                else if (std::is_same<dtype, std::uint32_t>::value)
                {
                    type = OnnxType::onnx_uint32;
                }
                else if (std::is_same<dtype, std::uint64_t>::value)
                {
                    type = OnnxType::onnx_uint64;
                }
            }
            Blob sliceSecondDimStep1(const int& start, const int& end)
            {
                if (shape[0] != 1)
                {
                    std::cout << "slice param not support!" << std::endl;
                    return Blob<dtype>();
                }
                if (start<0 || start>shape[1])
                {
                    std::cout << "slice param error!" << std::endl;
                    return Blob<dtype>();
                }
                int dataTotalCnt = std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<int>());
                int startPos = start * dataTotalCnt / shape[0] / shape[1];
                Blob<dtype> ret;
                ret.type = getType();
                ret.data = &data[startPos];
                ret.shape = shape;
                ret.shape[1] = (end <= shape[1] ? end : shape[1]) - start;
                return ret;
            }
            Blob flattenAxis2()
            {
                int totalCnt = std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<int>());
                int dim1 = 1;
                for (int i = 0; i < 2 && i < shape.size(); i++)
                {
                    dim1 *= shape[i];
                }
                int dim2 = totalCnt / dim1;
                Blob<dtype> ret;
                ret.data = data;
                ret.type = getType();
                ret.shape = cv::dnn::MatShape{ dim1,dim2 };
                return ret;
            }
            Blob reshape(const cv::dnn::MatShape& newShape)
            {
                int neg_1_count = 0;
                int neg_1_pos = -1;
                int multi = 1;
                for (int i = 0; i < newShape.size(); i++)
                {
                    if (newShape[i] == -1)
                    {
                        neg_1_count += 1;
                        neg_1_pos = i;
                    }
                    else
                    {
                        multi *= newShape[i];
                    }
                }
                if (neg_1_count > 1)
                {
                    std::cout << "slice param error!" << std::endl;
                    return Blob<dtype>();
                }
                cv::dnn::MatShape shape2 = newShape;
                if (neg_1_count == 1)
                {
                    shape2[neg_1_pos] = std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<int>()) / multi;
                }
                Blob<dtype> ret;
                ret.type = getType();
                ret.data = data;
                ret.shape = shape2;
                return ret;
            }
            bool needManualDestroy{ false };
            void operator=(const Blob<dtype>& other)
            {
                type = other.type;
                data = other.data;
                shape = other.shape;
                needManualDestroy = false;
            }
            Blob(const Blob<dtype>& other)
            {
                type = other.getType();
                data = other.data;
                shape = other.shape;
                needManualDestroy = false;
            }
            void cloneFrom(const Blob<dtype>& other)
            {
                type = other.type;
                shape = other.shape;
                int totalCnt = std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<int>());
                if (totalCnt > 0)
                {
                    data = new dtype[totalCnt];
                    memcpy(data, other.data, sizeof(dtype) * totalCnt);
                }
                needManualDestroy = true;
            }
            Blob unsqueezeFromTail(const int& cnt)
            {
                Blob<dtype> ret;
                ret.data = data;
                ret.shape = shape;
                ret.type = getType();
                for (size_t i = 0; i < cnt; i++)ret.shape.emplace_back(1);
                return ret;
            }
            Blob unsqueeze(const int& dim)
            {
                Blob<dtype> ret;
                ret.data = data;
                ret.shape = shape;
                ret.type = getType();
                ret.shape.insert(ret.shape.begin() + dim, 1);
                return ret;
            }
            Blob expandLike(const cv::dnn::MatShape& shapeLike)
            {
                const Blob& dataSrc = *this;
                const cv::dnn::MatShape& dataSrcShape = dataSrc.getShape();
                if (dataSrcShape.size() > shapeLike.size())
                {
                    std::cout << "slice param error!" << std::endl;
                    return Blob<dtype>();
                }
                for (size_t i = 0; i < dataSrcShape.size(); i++)
                {
                    if (dataSrcShape[i] != shapeLike[i])
                    {
                        std::cout << "slice param error!" << std::endl;
                        return Blob<dtype>();
                    }
                }
                int totalCnt = std::accumulate(shapeLike.begin(), shapeLike.end(), 1, std::multiplies<int>());
                Blob<dtype> ret;
                ret.type = getType();
                ret.getMutableShape() = shapeLike;
                ret.setOtherData(new dtype[totalCnt]);
                ret.needManualDestroy = true;
                int outerLoop = std::accumulate(dataSrcShape.begin(), dataSrcShape.end(), 1, std::multiplies<int>());
                int innerLoop = totalCnt / outerLoop;
                int idx = 0;
                for (int i = 0; i < outerLoop; i++)
                {
                    const dtype& d = dataSrc.getData()[i];
                    for (int j = 0; j < innerLoop; j++)
                    {
                        ret.getMutableData()[idx] = d;
                        idx += 1;
                    }
                }
                return ret;
            }
            Blob gatherDim0(const std::vector<int>& gatherAxis)
            {
                if (gatherAxis.size() == 0)
                {
                    std::cout << "gatherDim0 param error!" << std::endl;
                    return Blob<dtype>();
                }
                if (gatherAxis.size() == 1)
                {
                    int totalCnt = std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<int>());
                    int startPos = totalCnt / shape[0] * gatherAxis[0];
                    Blob<dtype> ret;
                    ret.type = getType();
                    ret.data = data + startPos;
                    ret.shape = shape;
                    ret.shape[0] = 1;
                    return ret;
                }
                else
                {
                    std::cout << "gatherDim0 param not support yet!" << std::endl;
                    return Blob<dtype>();
                }
            }
            Blob& whereInplace(const Blob& condition, const Blob& other)
            {
                if (shape == condition.getShape() && shape == other.getShape())
                {

                }
                else
                {
                    std::cout << "gatherDim0 param error!" << std::endl;
                    return *this;
                }
                int totalCnt = std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<int>());
                const dtype* conditionData = condition.getData();
                for (int i = 0; i < totalCnt; i++)
                {
                    if (conditionData[i] < 1)
                    {
                        data[i] = other.getData()[i];
                    }
                }
                return *this;
            }
            Blob& whereInplaceAndClip(const Blob& condition, const Blob& other, const dtype& clipMin, const dtype& clipMax)
            {
                if (shape == condition.getShape() && shape == other.getShape())
                {

                }
                else
                {
                    std::cout << "gatherDim0 param error!" << std::endl;
                    return *this;
                }
                int totalCnt = std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<int>());
                const dtype* conditionData = condition.getData();
                for (int i = 0; i < totalCnt; i++)
                {
                    if (conditionData[i] < 1)
                    {
                        data[i] = other.getData()[i];
                    }
                    if (data[i] < clipMin)
                    {
                        data[i] = clipMin;
                    }
                    else if (data[i] > clipMax)
                    {
                        data[i] = clipMax;
                    }
                }
                return *this;
            }
            cv::Mat convertToMat(const int& featHeight, const int& featWidth, const int& imgHeight, const int& imgWidth)
            {
                if (type == OnnxType::onnx_float32)
                {
                    cv::Mat featMask(featHeight, featWidth, CV_32FC1, data);
                    cv::Mat ret;
                    cv::resize(featMask, ret, cv::Size(imgWidth, imgHeight));
                    return ret;
                }
                else
                {
                    std::cout << "slice param not support!" << std::endl;
                    return cv::Mat();
                }
            }
            std::vector<dtype>convertToVec()
            {
                int dataTotalCnt = std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<int>());
                std::vector<dtype> ret(dataTotalCnt);
                memcpy(&ret[0], data, sizeof(dtype) * dataTotalCnt);
                return ret;
            }
            ~Blob()
            {
                if (needManualDestroy && data != nullptr)
                {
                    delete[]data;
                    data = nullptr;
                }
            }
        private:
            cv::dnn::MatShape shape;
            OnnxType type;
        public:
            dtype* data{ nullptr };
        };
        bool generDnnBlob(cv::Mat& blob, const cv::dnn::MatShape& shape, const OnnxType& type = OnnxType::onnx_float32);
        bool generPositionBlob(const std::vector<cv::Vec2f>& point_coord,
            const std::vector<float>& point_label,
            cv::Mat& point_coord_blob,
            cv::Mat& point_label_blob,
            const cv::Size& originalImgSize);
        bool convertNcnnBlobToOpencv(const ncnn::Mat& data, const std::vector<int>& targetShape, cv::Mat& out);
        bool serializationBlob(const cv::Mat& blob, cv::dnn::MatShape& shape, std::vector<float>& dat);
        cv::dnn::MatShape getBlobShape(const cv::Mat& blob);
        std::vector<int>getDenominators(const cv::dnn::MatShape& shape);
        std::vector<int>getPos(const int& idx, const std::vector<int>& denominators);
        void printBlob(const cv::Mat& blob);
        void printInt64Blob(const cv::Mat& blob);
        void checkSoftmax(const cv::Mat& blob);
        bool readBlobFile(const std::string& path, cv::Mat& blob, const std::optional<cv::dnn::MatShape>optionalShape = std::nullopt);
    }
}

#endif // !_DNN_HELPER
