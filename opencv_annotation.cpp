
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/imgproc.hpp>
#include "opencv2/dnn.hpp"
//#include "opencv2/calib3d.hpp"
#include <fstream>
#include <iostream>
#include <map>
#include <unordered_set>
#include <vector>
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
    const int& netImgSize = 1024)
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
        ((float*)point_coord_blob.data)[2 * i] = point_coord[i][0] / netImgSize;
        ((float*)point_coord_blob.data)[2 * i + 1] = point_coord[i][1] / netImgSize;
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
        std::cout << shape[i] << " ";
    }
    std::cout << std::endl;
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
int main(int argc, const char** argv)
{
    //return test_dynamic_reshape();
    const int netImgSize = 1024;
    //cv::dnn::Net imgEncoderNet = cv::dnn::readNetFromONNX("D:/repo/colmapThird/opencv_encoder.onnx");
    //int sz[] = { 1,3,netImgSize,netImgSize };
    //Mat imgBlob;
    //imgBlob.create(4, sz, CV_32F);
    //imgBlob.setTo(1); 
    //imgEncoderNet.setInput(imgBlob);
    //std::vector<cv::Mat> imgEncoderNetOut;
    //std::vector<std::string> outLayersNames = { "high_res_feats_0","high_res_feats_1","image_embed" };
    //imgEncoderNet.forward(imgEncoderNetOut, outLayersNames);  // crash here
    //std::cout << "forward ok " << std::endl;
    //cv::Mat outs(64 * 64, 128, CV_32FC1);
    //memcpy(outs.data, imgEncoderNetOut[0].data, 64 * 64 * 128 * sizeof(float));
    //getBlobShape(imgEncoderNetOut[0]);
    //getBlobShape(imgEncoderNetOut[1]);
    //getBlobShape(imgEncoderNetOut[2]);

    cv::Mat high_res_feats_0;
    cv::Mat high_res_feats_1;
    cv::Mat image_embed;
    generTestBlob(high_res_feats_0, { 1, 32, 256, 256 });
    generTestBlob(high_res_feats_1, { 1 ,64, 128, 128 });
    generTestBlob(image_embed, { 1, 256, 64, 64 });

    cv::Size originalImgSize(1920, 1080);
    cv::dnn::Net positionEmbedingNet = cv::dnn::readNetFromONNX("D:/repo/colmapThird/decoderBody2.onnx");
    std::vector<cv::Vec2f>point_coord = { {10., 10.} ,{500., 400.},{200., 600.},{100., 300.},{200., 300.},{1,1} };
    std::vector<float>point_label = { 1,1,1,1,-1 ,1 };

    cv::Mat point_coord_blob;
    cv::Mat point_label_blob;
    cv::Mat inputArrayPlus6;
    generPositionBlob(point_coord, point_label, point_coord_blob, point_label_blob, netImgSize);
    generTestBlob(inputArrayPlus6, { 1,static_cast<int>(point_coord.size()) + 6,1 });
    cv::Mat mask_input;
    generTestBlob(mask_input, { 1, 1, 1024 / 4, 1024 / 4 });
    mask_input.setTo(0);
    cv::Mat has_mask_input;
    generTestBlob(has_mask_input, { 1 });
    has_mask_input.setTo(1);
    cv::Mat orig_im_size;
    generTestBlob(orig_im_size, { 2 });
    ((float*)orig_im_size.data)[0] = originalImgSize.width;
    ((float*)orig_im_size.data)[1] = originalImgSize.height;
    {
        positionEmbedingNet.setInput(high_res_feats_0, "high_res_feats_0");
        positionEmbedingNet.setInput(high_res_feats_1, "high_res_feats_1");
        positionEmbedingNet.setInput(image_embed, "image_embed");
        positionEmbedingNet.setInput(point_coord_blob, "/ScatterND_1_output_0");
        positionEmbedingNet.setInput(inputArrayPlus6, "inputArrayPlus6");
        positionEmbedingNet.setInput(point_label_blob, "/Unsqueeze_8_output_0");
        positionEmbedingNet.setInput(mask_input, "mask_input");
        positionEmbedingNet.setInput(has_mask_input, "has_mask_input");
        positionEmbedingNet.setInput(orig_im_size, "orig_im_size");
        std::vector<std::string> layersNames = positionEmbedingNet.getLayerNames();
        std::vector<std::string> unconnectedOutLayersNames = positionEmbedingNet.getUnconnectedOutLayersNames();
        std::vector<std::string> outLayersNames = {
            "/transformer/final_attn_token_to_image/MatMul_1_output_0"
        };
        std::vector<cv::Mat> out;
        positionEmbedingNet.forward(out, outLayersNames);
        printBlob(out[0]);
        printBlob(out[1]);
        //checkSoftmax(out[1]);
        printBlob(out[2]);
        printBlob(out[3]);
        printBlob(out[4]);
        std::cout << "forward ok " << std::endl;
    }

    cv::dnn::Net pointLabelsInNet = cv::dnn::readNetFromONNX("D:/repo/colmapThird/pointLabelsIn.onnx");

    std::string paramPath = "D:/repo/colmapThird/positionEmbeding.onnx";
    std::cout << 123 << std::endl;
    cv::dnn::Net net = cv::dnn::readNetFromONNX(paramPath);




    return 0;
}
