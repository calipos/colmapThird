
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/imgproc.hpp>
#include "opencv2/dnn.hpp"
//#include "opencv2/calib3d.hpp"
#include <fstream>
#include <numeric>
#include <iostream>
#include <map>
#include <unordered_set>
#include <vector>
#include "opencv2/dnn/dnn.hpp"
#include "opencv2/dnn/layer.hpp"
#include "opencv2/dnn/shape_utils.hpp"
using namespace std;
using namespace cv;

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
bool operator==(const cv::dnn::MatShape& shape1, const cv::dnn::MatShape& shape2)
{
    if (shape1.size() != shape2.size())
    {
        return false;
    }
    for (int i = 0; i < shape1.size(); i++)
    {
        if (shape1[i] != shape2[i])
        {
            return false;
        }
    }
    return true;
}
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
    bool setOtherData(dtype*otherData)
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
        if( shape[0]!=1)
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
        int startPos = start*dataTotalCnt / shape[0] / shape[1];
        Blob<dtype> ret;
        ret.type = getType() ;
        ret.data = &data[startPos];
        ret.shape = shape;
        ret.shape[1] = (end<= shape[1]? end: shape[1]) - start;
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
        int dim2 = totalCnt/dim1;
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
        if (neg_1_count>1)
        {
            std::cout << "slice param error!" << std::endl;
            return Blob<dtype>();
        }
        cv::dnn::MatShape shape2 = newShape;
        if (neg_1_count==1)
        {
            shape2[neg_1_pos] = std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<int>()) /multi;
        }
        Blob<dtype> ret;
        ret.type = getType();
        ret.data = data;
        ret.shape = shape2;
        return ret;
    }
    bool needManualDestroy{ false };
    void operator=(const Blob<dtype>&other)
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
        if (totalCnt>0)
        { 
            data = new dtype[totalCnt];
            memcpy(data, other.data,sizeof(dtype)* totalCnt);
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
        ret.shape.insert(ret.shape.begin() + dim,1);
        return ret;
    }
    Blob expandLike(const cv::dnn::MatShape&shapeLike)
    {
        const Blob& dataSrc = *this;
        const cv::dnn::MatShape& dataSrcShape = dataSrc.getShape();
        if (dataSrcShape.size()> shapeLike.size())
        {
            std::cout << "slice param error!" << std::endl;
            return Blob<dtype>();
        }
        for (size_t i = 0; i < dataSrcShape.size(); i++)
        {
            if (dataSrcShape[i]!= shapeLike[i])
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
        int innerLoop = totalCnt/ outerLoop;
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
    Blob gatherDim0(const std::vector<int>&gatherAxis)
    {
        if (gatherAxis.size()==0)
        {
            std::cout << "gatherDim0 param error!" << std::endl;
            return Blob<dtype>();
        }
        if (gatherAxis.size()==1)
        {
            int totalCnt = std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<int>());
            int startPos = totalCnt/shape[0]*gatherAxis[0];
            Blob<dtype> ret;
            ret.type = getType();
            ret.data = data+ startPos;
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
bool generTestBlob(cv::Mat& blob, const cv::dnn::MatShape& shape, const OnnxType& type = OnnxType::onnx_float32)
{
    switch (type)
    {
    case OnnxType::onnx_float32:
        blob.create(shape.size(), &shape[0], CV_32F);
        break;
    case OnnxType::onnx_int64:
        blob.create(shape.size(), &shape[0], CV_64FC1);
        break;
    case OnnxType::onnx_int32:
        blob.create(shape.size(), &shape[0], CV_32SC1);
        break;
    default:
        break;
    }
    blob.setTo(1);
    return true;
}
bool generPositionBlob(const std::vector<cv::Vec2f>& point_coord,
    const std::vector<float>& point_label,
    cv::Mat& point_coord_blob,
    cv::Mat& point_label_blob,
    const cv::Size& originalImgSize)
{
    if (point_coord.size() != point_label.size())
    {
        return false;
    }
    int sz1[] = { 1,point_coord.size() + 1,2 };
    point_coord_blob.create(3, sz1, CV_32F);
    for (size_t i = 0; i < point_coord.size() + 1; i++)
    {
        if (i >= point_coord.size())
        {
            ((float*)point_coord_blob.data)[2 * i] = 0;
            ((float*)point_coord_blob.data)[2 * i + 1] = 0;
            break;
        }
        ((float*)point_coord_blob.data)[2 * i] = point_coord[i][0] / originalImgSize.width;
        ((float*)point_coord_blob.data)[2 * i + 1] = point_coord[i][1] / originalImgSize.height;
    }
    int sz2[] = { 1,point_coord.size() + 1,1 };
    point_label_blob.create(3, sz2, CV_32F);
    for (size_t i = 0; i < point_coord.size() + 1; i++)
    {
        if (i >= point_coord.size())
        {
            ((float*)point_label_blob.data)[i] = -1;
            break;
        }
        if (point_label[i] > 0)
        {
            ((float*)point_label_blob.data)[i] = 1;
        }
        else
        {
            ((float*)point_label_blob.data)[i] = -1;
        }
    }
    return true;
}
cv::dnn::MatShape getBlobShape(const cv::Mat& blob)
{
    const int* shape = blob.size.p;
    const int& dim = blob.dims;
    cv::dnn::MatShape shapeRet(dim);
    for (size_t i = 0; i < dim; i++)
    {
        shapeRet[i] = shape[i];
    }
    return shapeRet;
}
std::vector<int>getDenominators(const cv::dnn::MatShape& shape)
{
    std::vector<int>denominators(shape.size(), 1);
    for (int i = shape.size() - 2; i >= 0; i--)
    {
        denominators[i] = denominators[i + 1] * shape[i + 1];
    }
    denominators.back() = 1;
    return denominators;
}
std::vector<int>getPos(const int& idx, const std::vector<int>& denominators)
{
    std::vector<int>pos(denominators.size(), 0);
    pos[0] = idx;
    for (size_t i = 0; i < denominators.size(); i++)
    {
        if (i + 1 < denominators.size())
        {
            pos[i + 1] = pos[i] % denominators[i];
        }
        pos[i] /= denominators[i];
    }
    return pos;
}
void printBlob(const cv::Mat& blob)
{
    cv::dnn::MatShape shape = getBlobShape(blob);
    std::vector<int>denominators = getDenominators(shape);
    int total = 1;
    for (int c = 0; c < shape.size(); c++)
    {
        total *= shape[c];
    }
    int lineCnt = shape.back();
    bool slashN = true;
    int dotCount = 0;
    int dotCountMax = 10;
    for (int i = 0; i < total; i++)
    {
        if (i % lineCnt == 0 && i > 0 && slashN)
        {
            std::cout << std::endl;
        }
        std::vector<int>pos = getPos(i, denominators);
        int showFlag = 0;
        for (size_t j = 0; j < shape.size(); j++)
        {
            if (shape[j] > 10 && (pos[j] == 4 || pos[j] == (shape[j] - 4)))
            {
                showFlag = 1;
                break;
            }
            if (shape[j] > 10 && pos[j] > 4 && pos[j] < (shape[j] - 4))
            {
                showFlag = 2;
                break;
            }
        }
        if (showFlag == 0)
        {
            dotCount = 0;
            std::cout << ((float*)blob.data)[i] << " ";
            slashN = true;
        }
        else if (showFlag == 1)
        {
            if (dotCount< dotCountMax)
            {
                std::cout << " ... ";
                dotCount += 1;
            }
            slashN = true;
        }
        else if (showFlag == 2)
        {
            slashN = false;
        }
    }
    std::cout << std::endl;
    return;
}
void printInt64Blob(const cv::Mat& blob)
{
    cv::dnn::MatShape shape = getBlobShape(blob);
    int total = 1;
    for (int c = 0; c < shape.size(); c++)
    {
        total *= shape[c];
    }
    int slashN = shape.back();
    for (int i = 0; i < total; i++)
    {
        if (i % slashN == 0)
        {
            std::cout << std::endl;
        }
        std::cout << ((std::int64_t*)blob.data)[i] << " ";
    }
    std::cout << std::endl;
    return;
}
void checkSoftmax(const cv::Mat& blob)
{
    cv::dnn::MatShape shape = getBlobShape(blob);
    int dim0 = 1;
    int dim1 = shape.back();
    for (int i = 0; i < shape.size() - 1; i++)
    {
        dim0 *= shape[i];
    }
    std::vector<double>checkVALUE(dim0, 0);
    int idx = 0;
    const float* data = (const float*)blob.data;
    for (int i = 0; i < dim0; i++)
    {
        for (int j = 0; j < dim1; j++)
        {
            checkVALUE[i] += data[idx];
            idx++;
        }
    }


    return;
}
int test_dynamic_reshape()
{
    cv::Mat input;
    generTestBlob(input, { 1, 3,2 });
    float* p = (float*)input.data;
    p[0] = 1;
    p[1] = 2;
    p[2] = 3;
    p[3] = 4;
    p[4] = 5;
    p[5] = 6;
    cv::dnn::Net testNet = cv::dnn::readNetFromONNX("D:/repo/colmapThird/test.onnx");
    testNet.setInput(input, "input");
    std::vector<std::string> layersNames = testNet.getLayerNames();
    std::vector<std::string> unconnectedOutLayersNames = testNet.getUnconnectedOutLayersNames();
    std::vector<std::string> outLayersNames = {
    "output","outputReshape" };
    std::vector<cv::Mat> out;
    testNet.forward(out, outLayersNames);
    printBlob(out[0]);
    printBlob(out[1]);
    return 0;
}

//"/Reshape_12_output_0", "/GreaterOrEqual_output_0","/iou_prediction_head/Sigmoid_output_0",  "/ArgMax_output_0"
int decoderTails(const int& originalImgHeight, const int& originalImgWidth,
    const cv::Mat& Reshape_12_output_0_, const cv::Mat& GreaterOrEqual_output_0_, const cv::Mat& iou_prediction_head_Sigmoid_output_0_, const cv::Mat& ArgMax_output_0_,cv::Mat& mask,std::vector<float>&iou_predictions)
{
    Blob<float> Reshape_12_output_0(Reshape_12_output_0_);
    Blob<float> GreaterOrEqual_output_0(GreaterOrEqual_output_0_);
    Blob<float> iou_prediction_head_Sigmoid_output_0(iou_prediction_head_Sigmoid_output_0_);
    Blob<float> ArgMax_output_0(ArgMax_output_0_);//onnx out int64,but in opencv the data is float still
    auto Slice_8_output_0 = Reshape_12_output_0.sliceSecondDimStep1(0,1);
    const cv::dnn::MatShape& Shape_37 = Slice_8_output_0.getShape();
    auto Expand_10_output_0 = GreaterOrEqual_output_0.expandLike(Shape_37);


    auto Slice_6_output_0 = Reshape_12_output_0.sliceSecondDimStep1(1, std::numeric_limits<int>::max());
    const auto& Shape_33_output_0 = Slice_6_output_0.getShape();
    const int& Gather_26_output_0 = Shape_33_output_0[1];

    auto Slice_7_output_0 = iou_prediction_head_Sigmoid_output_0.sliceSecondDimStep1(1, std::numeric_limits<int>::max());
    auto Slice_9_output_0 = iou_prediction_head_Sigmoid_output_0.sliceSecondDimStep1(0, 1);
    auto Shape_39_output_0 = Slice_9_output_0.getShape();
    auto Expand_11_output_0 = GreaterOrEqual_output_0.expandLike(Shape_39_output_0);
    

    const auto& Shape_32_output_0 = Slice_7_output_0.getShape();
    const int& Gather_25_output_0 = Shape_32_output_0[0];
    Blob<float>Flatten_1_output_0 = Slice_7_output_0.flattenAxis2();
    std::vector<int>Add_14_output_0; Add_14_output_0.reserve(4);
    std::vector<int>Add_15_output_0; Add_15_output_0.reserve(4);
    std::vector<int>Concat_19_output_0; Concat_19_output_0.reserve(4);
    for (int i = 0; i < Gather_25_output_0; i++)
    {
        Add_15_output_0.emplace_back(i * 3 + ArgMax_output_0.getData()[0]);
        int d = i * Gather_26_output_0 + ArgMax_output_0.getData()[0];
        Add_14_output_0.emplace_back(d);
        Concat_19_output_0.emplace_back(d);
    }
    Concat_19_output_0.emplace_back(Shape_33_output_0[3]);
    Concat_19_output_0.emplace_back(Shape_33_output_0[3]);
    Blob<float>Flatten_output_0 = Slice_6_output_0.flattenAxis2();
    Blob<float>Gather_29_output_0 = Flatten_output_0.gatherDim0(Add_14_output_0);
    Blob<float>Reshape_14_output_0 = Gather_29_output_0.reshape(Concat_19_output_0);
    Blob<float>Unsqueeze_26_output_0 = Reshape_14_output_0.unsqueeze(1);
    auto& Where_8_output_0 = Slice_8_output_0.whereInplaceAndClip(Expand_10_output_0, Unsqueeze_26_output_0,-32,32);
    mask = Where_8_output_0.convertToMat(256, 256, originalImgHeight, originalImgWidth);

    Blob<float>Gather_31_output_0 = Flatten_1_output_0.gatherDim0(Add_15_output_0);
    Blob<float> Unsqueeze_27_output_0 = Gather_31_output_0.reshape(Shape_39_output_0);
    auto& iou_predictions_blob = Slice_9_output_0.whereInplace(Expand_11_output_0, Unsqueeze_27_output_0);
    iou_predictions = iou_predictions_blob.convertToVec();
    return 0;
}
int main(int argc, const char** argv)
{ 

    cv::Mat img = cv::imread("D:/ucl360/UCL360Calib/CameraCalibGui/pro45/out/1_-180.bmp");
    cv::Size originalImgSize= img.size();
    cv::Scalar mean(0.485 * 255, 0.456 * 255, 0.406 * 255);
    cv::Mat imgBlob = cv::dnn::blobFromImage(img, 1. / 57.12, cv::Size(1024, 1024), mean, true);
    //return test_dynamic_reshape();
    const int netImgSize = 1024;
    cv::dnn::Net imgEncoderNet = cv::dnn::readNetFromONNX("D:/repo/colmapThird/models/opencv_encoder.onnx");
    int sz[] = { 1,3,netImgSize,netImgSize };

    imgEncoderNet.setPreferableBackend(cv::dnn::DNN_TARGET_CPU);
    imgEncoderNet.setInput(imgBlob);
    std::vector<cv::Mat> imgEncoderNetOut;
    std::vector<std::string> outLayersNames = { "high_res_feats_0","high_res_feats_1","image_embed" };
    imgEncoderNet.forward(imgEncoderNetOut, outLayersNames);  // crash here
    std::cout << "encode forward ok " << std::endl;


    cv::Mat& high_res_feats_0 = imgEncoderNetOut[0];
    cv::Mat& high_res_feats_1 = imgEncoderNetOut[1];
    cv::Mat& image_embed = imgEncoderNetOut[2];


    cv::dnn::Net positionDecoderNet = cv::dnn::readNetFromONNX("D:/repo/colmapThird/models/opencv_decoder.onnx");
    //std::vector<cv::Vec2f>point_coord = { {10., 10.} ,{500., 400.},{200., 600.},{100., 300.},{200., 300.},{1,1} };
    //std::vector<float>point_label = { 1,1,1,1,-1 ,1 };
    std::vector<cv::Vec2f>point_coord = { {1161,301} };
    std::vector<float>point_label = { 1 };

    positionDecoderNet.setPreferableBackend(cv::dnn::DNN_TARGET_CPU);
    cv::Mat point_coord_blob;
    cv::Mat point_label_blob;
    cv::Mat inputArrayPlus6;
    generPositionBlob(point_coord, point_label, point_coord_blob, point_label_blob, originalImgSize);
    generTestBlob(inputArrayPlus6, { 1,static_cast<int>(point_coord.size()) + 6,1 });
    cv::Mat mask_input;
    generTestBlob(mask_input, { 1, 1, 1024 / 4, 1024 / 4 });
    mask_input.setTo(0);
    cv::Mat has_mask_input;
    generTestBlob(has_mask_input, { 1 });
    has_mask_input.setTo(1);
    cv::Mat orig_im_size;
    generTestBlob(orig_im_size, { 2 }, OnnxType::onnx_int32);
    ((int*)orig_im_size.data)[0] = originalImgSize.width;
    ((int*)orig_im_size.data)[1] = originalImgSize.height;
    cv::Mat mask;
    std::vector<float> iou_predictions;
    {
        positionDecoderNet.setInput(high_res_feats_0, "high_res_feats_0");
        positionDecoderNet.setInput(high_res_feats_1, "high_res_feats_1");
        positionDecoderNet.setInput(image_embed, "image_embed");
        positionDecoderNet.setInput(point_coord_blob, "/ScatterND_1_output_0");
        positionDecoderNet.setInput(inputArrayPlus6, "inputArrayPlus6");
        positionDecoderNet.setInput(point_label_blob, "/Unsqueeze_8_output_0");
        positionDecoderNet.setInput(mask_input, "mask_input");
        positionDecoderNet.setInput(has_mask_input, "has_mask_input");
        //positionDecoderNet.setInput(orig_im_size, "orig_im_size");
        std::vector<std::string> layersNames = positionDecoderNet.getLayerNames();
        std::vector<std::string> unconnectedOutLayersNames = positionDecoderNet.getUnconnectedOutLayersNames();
        std::vector<std::string> outLayersNames = {
                "/Reshape_12_output_0","/GreaterOrEqual_output_0","/iou_prediction_head/Sigmoid_output_0","/ArgMax_output_0"
        };
        std::vector<cv::Mat> out;
        positionDecoderNet.forward(out, outLayersNames);
        //printBlob(out[0]);
        //printBlob(out[1]);
        //printBlob(out[2]);
        //printBlob(out[3]);
        std::cout << "forward ok " << std::endl;
        decoderTails(1080,1920, out[0], out[1], out[2], out[3], mask, iou_predictions);
        std::cout << "done " << std::endl;

        cv::Mat asd2;
        cv::threshold(mask, asd2, 0, 255, cv::THRESH_BINARY);
        asd2.convertTo(asd2, CV_8UC1);
        cv::imwrite("D:/repo/colmapThird/mask.png",asd2);
    }


    return 0;
}
