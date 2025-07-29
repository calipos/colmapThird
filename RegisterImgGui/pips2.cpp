#include "pips2.h"
#include "opencv2/opencv.hpp"
#include <fstream>
#include <numeric>
#include <iostream>
#include <map>
#include <memory>
#include <unordered_set>
#include <chrono>
#include <vector>
#include "dnnHelper.h"
#include "opencv2/dnn/dnn.hpp"
#include "opencv2/dnn/layer.hpp"
#include "opencv2/dnn/shape_utils.hpp"
#include "net.h"
#include "mat.h"
#include "colString.h"
namespace pips2
{
    std::string getBilinearOpNet()
    {
        std::string paramStr = "7767517\n"
            "15 15\n"
            "Input      v_y0_x0         0 1 v_y0_x0\n"
            "Input      v_y0_x1         0 1 v_y0_x1\n"
            "Input      v_y1_x0         0 1 v_y1_x0\n"
            "Input      v_y1_x1         0 1 v_y1_x1\n"
            "Input      w_y0_x0         0 1 w_y0_x0\n"
            "Input      w_y0_x1         0 1 w_y0_x1\n"
            "Input      w_y1_x0         0 1 w_y1_x0\n"
            "Input      w_y1_x1         0 1 w_y1_x1\n"
            "MatMul      m1            2 1 v_y0_x0 w_y0_x0 m1\n"
            "MatMul      m2            2 1 v_y0_x1 w_y0_x1 m2\n"
            "MatMul      m3            2 1 v_y1_x0 w_y1_x0 m3\n"
            "MatMul      m4            2 1 v_y1_x1 w_y1_x1 m4\n"
            "BinaryOp     a1            2 1 m1 m2 a1 0=0\n"
            "BinaryOp     a2            2 1 m3 m4 a2 0=0\n"
            "BinaryOp     output          2 1 a1 a2 output 0=0\n";
        return paramStr;
    }
    ncnn::Mat bilinear_sample2d(const ncnn::Mat& blob, const std::vector<float>& xs, const std::vector<float>& ys, std::shared_ptr<ncnn::Net> bilinearOpNet)
    {
        int C = blob.c;
        int N = xs.size();
        int W = blob.w;
        float W_f = blob.w;
        float H_f = blob.h;
        int max_x = blob.w - 1;
        int max_y = blob.h - 1;
        ncnn::Mat v_y0_x0(N, C, (size_t)4);
        ncnn::Mat v_y0_x1(N, C, (size_t)4);
        ncnn::Mat v_y1_x0(N, C, (size_t)4);
        ncnn::Mat v_y1_x1(N, C, (size_t)4);
        ncnn::Mat w_y0_x0(1, N, (size_t)4);
        ncnn::Mat w_y0_x1(1, N, (size_t)4);
        ncnn::Mat w_y1_x0(1, N, (size_t)4);
        ncnn::Mat w_y1_x1(1, N, (size_t)4);
        for (size_t i = 0; i < N; i++)
        {
            int x0 = std::floor(xs[i]);
            int x1 = x0 + 1;
            int y0 = std::floor(ys[i]);
            int y1 = y0 + 1;
            if (x0 < 0)x0 = 0;
            if (y0 < 0)y0 = 0;
            if (x0 > max_x)x0 = max_x;
            if (y0 > max_y)y0 = max_y;
            if (x1 < 1)x1 = 1;
            if (y1 < 1)y1 = 1;
            if (x1 > max_x)x1 = max_x;
            if (y1 > max_y)y1 = max_y;

            ((float*)w_y0_x0.data)[i] = (x1 - xs[i]) * (y1 - ys[i]);
            ((float*)w_y0_x1.data)[i] = (xs[i] - x0) * (y1 - ys[i]);
            ((float*)w_y1_x0.data)[i] = (x1 - xs[i]) * (ys[i] - y0);
            ((float*)w_y1_x1.data)[i] = (xs[i] - x0) * (ys[i] - y0);

            for (int c = 0; c < C; c++)
            {
                int pp = i + c* v_y0_x0.w;
                int p0 = x0 + W * y0;
                int p1 = x1 + W * y0;
                int p2 = x0 + W * y1;
                int p3 = x1 + W * y1;
                ((float*)v_y0_x0.data)[pp] = ((const float*)blob.data)[p0 + c * blob.cstep];
                ((float*)v_y0_x1.data)[pp] = ((const float*)blob.data)[p1 + c * blob.cstep];
                ((float*)v_y1_x0.data)[pp] = ((const float*)blob.data)[p2 + c * blob.cstep];
                ((float*)v_y1_x1.data)[pp] = ((const float*)blob.data)[p3 + c * blob.cstep];
            }
        }

        ncnn::Extractor ex2 = bilinearOpNet->create_extractor();
        ex2.input("v_y0_x0", v_y0_x0);
        ex2.input("v_y0_x1", v_y0_x1);
        ex2.input("v_y1_x0", v_y1_x0);
        ex2.input("v_y1_x1", v_y1_x1);
        ex2.input("w_y0_x0", w_y0_x0);
        ex2.input("w_y0_x1", w_y0_x1);
        ex2.input("w_y1_x0", w_y1_x0);
        ex2.input("w_y1_x1", w_y1_x1);
        auto start1 = std::chrono::steady_clock::now();

        ncnn::Mat bilinear_sample_out;
        ex2.extract("output", bilinear_sample_out);
        auto end1 = std::chrono::steady_clock::now();
        auto elapsed1 = std::chrono::duration_cast<std::chrono::milliseconds>(end1 - start1).count();
        dnn::ncnnHelper::printBlob(bilinear_sample_out);
        std::cout << "Elapsed time: " << elapsed1 << " ms" << std::endl;

        return bilinear_sample_out;
    }

    bool Pips2::changeParamResizeParam(const std::string& path, const std::pair<int, int>& d)
    {
        if (!std::filesystem::exists(path))
        {
            return false;
        }
        std::list<std::string>alines;
        std::fstream fin(path, std::ios::in);
        std::string aline;
        while (std::getline(fin, aline))
        {
            if (aline.length() > 7)
            {
                std::string sub_sv = aline.substr(0, 6);
                if (sub_sv.compare("Interp") == 0)
                {
                    std::vector<std::string> segs = splitString(aline, " ", true);
                    std::string newLine;
                    for (int i = 0; i < segs.size(); i++)
                    {
                        if (segs[i].length() > 2 && segs[i][0] == '3' && segs[i][1] == '=')
                        {
                            newLine += "3=";
                            newLine += std::to_string(d.first);
                            newLine += " ";
                        }
                        else if (segs[i].length() > 2 && segs[i][0] == '4' && segs[i][1] == '=')
                        {
                            newLine += "4=";
                            newLine += std::to_string(d.second);
                            newLine += " ";
                        }
                        else
                        {
                            newLine += segs[i];
                            newLine += " ";
                        }
                    }
                    alines.emplace_back(newLine);
                }
                else
                {
                    alines.emplace_back(aline);
                }
            }
            else
            {
                alines.emplace_back(aline);
            }
        }
        fin.close();
        std::fstream fout(path, std::ios::out);
        for (const auto& d : alines)
        {
            fout << d << std::endl;
        }
        fout.close();
        return true;
    }
    Pips2::Pips2(
        const std::filesystem::path& ncnnEncoderParamPath, const std::filesystem::path& ncnnEncoderBinPath)
    {
        imgSize.width = -1;
        if (!std::filesystem::exists(ncnnEncoderParamPath))
        {
            LOG_ERR_OUT << "not found : " << ncnnEncoderParamPath;
            return;
        }
        if (!std::filesystem::exists(ncnnEncoderBinPath))
        {
            LOG_ERR_OUT << "not found : " << ncnnEncoderBinPath;
            return;
        }
        ncnnEncoderParamPath_ = ncnnEncoderParamPath;
        ncnnEncoderBinPath_ = ncnnEncoderBinPath;
    }
    Pips2::~Pips2()
    {}

    bool Pips2::inputImage(const std::vector<std::string>& imgPath,std::vector<ncnn::Mat>&fmaps)
    {
        if (imgPath.size()==0)
        {
            LOG_ERR_OUT << "imgPath.size()==0";
            return false;
        }
        fmaps.clear();
        fmaps.reserve(imgPath.size());
        for (int i = 0; i < imgPath.size(); i++)
        {
            if (!std::filesystem::exists(imgPath[i]))
            { 
                LOG_ERR_OUT << "imgPath.size()==0";
                return false;
            }
            cv::Mat img = cv::imread(imgPath[i]);
            if (imgSize.width<0)
            {
                imgSize.width = img.cols;
                imgSize.height = img.rows;
                //if (imgSize.width % 8 != 0 || imgSize.height % 8 != 0)
                //{
                //    LOG_ERR_OUT << "imgSize.width % 8 != 0 || imgSize.height % 8 != 0";
                //    return false;
                //}
                changeParamResizeParam(ncnnEncoderParamPath_.string(), std::make_pair(imgSize.height / 8, imgSize.width / 8));
                encoderNet.opt.use_vulkan_compute = true;
                encoderNet.opt.num_threads = 8;
                if (encoderNet.load_param(ncnnEncoderParamPath_.string().c_str()))
                    exit(-1);
                if (encoderNet.load_model(ncnnEncoderBinPath_.string().c_str()))
                    exit(-1);
                encoderNet.opt.blob_allocator;
                encoderNet.opt.workspace_allocator;
            }
            if (imgSize.width != img.cols || imgSize.height != img.rows)
            {

                LOG_ERR_OUT << "imgSize.width != img.cols || imgSize.height != img.rows";
                return false;
            }
            ncnn::Mat fmap;
            bool encoderRet = inputImage(img, fmap);
            if (!encoderRet)
            {
                LOG_ERR_OUT << "encoder fail.";
                return false;
            }
            fmaps.emplace_back(fmap);
        }
        return true;
    }
    bool Pips2::inputImage(const cv::Mat& img, ncnn::Mat& fmap)
    {
        ncnn::Mat in = ncnn::Mat::from_pixels(img.data, ncnn::Mat::PIXEL_BGR, img.cols, img.rows);
        ncnn::Extractor ex1 = encoderNet.create_extractor();
        ex1.input("rgbs", in);
        auto start1 = std::chrono::steady_clock::now();
        ex1.extract("fmaps", fmap);
        auto end1 = std::chrono::steady_clock::now();
        auto elapsed1 = std::chrono::duration_cast<std::chrono::milliseconds>(end1 - start1).count();
        //dnn::ncnnHelper::printBlob(fmap);
        std::cout << "Elapsed time: " << elapsed1 << " ms" << std::endl;
        return true;
    }
}



int test_pips2_ocv()
{


    cv::Mat inputBlob;
    dnn::ocvHelper::generDnnBlob(inputBlob, { 8, 3, 1024 , 1024 });
    inputBlob.setTo(1);
    cv::dnn::Net Net = cv::dnn::readNetFromONNX("../models/pips2_base_opencv.onnx");
    std::cout << "load EncoderNet." << std::endl; 
    Net.setPreferableBackend(cv::dnn::DNN_TARGET_CPU);
    Net.setInput(inputBlob);
    std::vector<cv::Mat> imgEncoderNetOut;
    std::vector<std::string> outLayersNames = { "fmaps" };
    auto start1 = std::chrono::steady_clock::now();
    Net.forward(imgEncoderNetOut, outLayersNames);  // crash here
    auto end1 = std::chrono::steady_clock::now();
    auto elapsed1 = std::chrono::duration_cast<std::chrono::milliseconds>(end1 - start1).count();
    std::cout << "Elapsed time: " << elapsed1 << " ms" << std::endl;
    //dnn::ocvHelper::printBlob(imgEncoderNetOut[0]);
	return 0;
}
int test_bilinearOp()
{
    std::string paramStr = pips2::getBilinearOpNet();
    std::shared_ptr<ncnn::Net> bilinearOpNet(new ncnn::Net());
    bilinearOpNet->load_param_mem(paramStr.c_str());


    std::vector<int>shape={128,64,64};
    int totalcnt = std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<int>());
    std::vector<float> indata(totalcnt);
    for (int i = 0; i < indata.size(); i++)
    {
        indata[i] = i%200-100;
    }
    ncnn::Mat in(shape[2], shape[1], shape[0], (void*)&indata[0], 4);
    std::vector<float>xs = { 12.5000, 13.3000, 23.3000 };
    std::vector<float>ys = { 1.2000, 45.1000, 15.1000 };

    pips2::bilinear_sample2d(in,xs,ys, bilinearOpNet);

    return 0;
}
int test_pips2()
{
    return test_bilinearOp();

    std::vector<std::string>paths = {
    "D:/repo/colmapThird/data2/a/00000.jpg", 
    "D:/repo/colmapThird/data2/a/00001.jpg", 
    "D:/repo/colmapThird/data2/a/00002.jpg", 
    "D:/repo/colmapThird/data2/a/00003.jpg", 
    "D:/repo/colmapThird/data2/a/00004.jpg", 
    "D:/repo/colmapThird/data2/a/00005.jpg", 
    "D:/repo/colmapThird/data2/a/00006.jpg", 
    "D:/repo/colmapThird/data2/a/00007.jpg", };

    pips2::Pips2 ins("../models/pips2_base_ncnn.param", "../models/pips2_base_ncnn.bin");
    std::vector<ncnn::Mat> fmaps;
    ins.inputImage(paths, fmaps);
    return 0;
}