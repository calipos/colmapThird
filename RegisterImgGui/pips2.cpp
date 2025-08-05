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
    const int Pips2::stride = 8;
    const int Pips2::corrsBlockCnt = 3;
    const int Pips2::latent_dim = 128;
    const int Pips2::pyramid_level = 4;
    const int Pips2::omega_temperature = 10000;


    std::string Pips2::getBilinearOpNet()
    {
        std::string paramStr = "7767517\n"
            "3 3\n"
            "Input      v         0 1 v\n"
            "Input      w         0 1 w\n"
            "MatMul      output            2 1 v w output\n";
        return paramStr;
    }

    std::string Pips2::getCorrsNet(const int&sequenceLength, const int& imgHeight, const int& imgWidth)
    {
        std::string paramStr =
            "7767517\n"
            "25 35\n"
            "Input            fmaps                    0 1 fmaps\n"
            "Split            splitncnn_input0         1 2 fmaps fmaps_splitncnn_0 fmaps_splitncnn_1\n"
            "Input            feats                    0 1 feats\n"
            "Split            splitncnn_input1         1 4 feats feats_splitncnn_0 feats_splitncnn_1 feats_splitncnn_2 feats_splitncnn_3\n"
            "Pooling          /AveragePool             1 1 fmaps_splitncnn_1 /AveragePool_output_0 0=1 1=2 11=2 2=2 12=2 3=0 13=0 14=0 15=0 5=1 6=1\n"  // 13=0 14=0 15=0 5=1 6=1
            "Split            splitncnn_0              1 2 /AveragePool_output_0 /AveragePool_output_0_splitncnn_0 /AveragePool_output_0_splitncnn_1\n"
            "Pooling          /AveragePool_1           1 1 /AveragePool_output_0_splitncnn_1 /AveragePool_1_output_0 0=1 1=2 11=2 2=2 12=2 3=0 13=0 14=0 15=0 5=1 6=1\n"
            "Split            splitncnn_1              1 2 /AveragePool_1_output_0 /AveragePool_1_output_0_splitncnn_0 /AveragePool_1_output_0_splitncnn_1\n"
            "Pooling          /AveragePool_2           1 1 /AveragePool_1_output_0_splitncnn_1 /AveragePool_2_output_0 0=1 1=2 11=2 2=2 12=2 3=0 13=0 14=0 15=0 5=1 6=1\n";
        int h0 = imgHeight / Pips2::stride;
        int w0 = imgWidth / Pips2::stride;
        std::string reshape0Str = "Reshape /Reshape 1 1 fmaps_splitncnn_0 /Reshape_output_0 2=" + std::to_string(sequenceLength) + " 1=" + std::to_string(Pips2::latent_dim) + " 0=" + std::to_string(h0 * w0) + "\n";
        paramStr += reshape0Str;
        paramStr +="MatMul           /MatMul                  2 1 feats_splitncnn_3 /Reshape_output_0 /MatMul_output_0\n";
        std::string div0Str ="BinaryOp  /Div  1 1 /MatMul_output_0 corrs_pyramid_reshape_0 0=3  1=1  2="+std::to_string(sqrt(double(Pips2::latent_dim)))+"\n";
        paramStr += div0Str;
        int h1 = h0 / 2;
        int w1 = w0 / 2;
        std::string reshape1Str = "Reshape /Reshape_1 1 1 /AveragePool_output_0_splitncnn_0 /Reshape_1_output_0 2=" + std::to_string(sequenceLength) + " 1=" + std::to_string(Pips2::latent_dim) + " 0=" + std::to_string(h1 * w1) + "\n";
        paramStr += reshape1Str;
        paramStr +="MatMul           /MatMul_1                2 1 feats_splitncnn_2 /Reshape_1_output_0 /MatMul_1_output_0\n";
        std::string div1Str = "BinaryOp  /Div_1  1 1 /MatMul_1_output_0 corrs_pyramid_reshape_1 0=3  1=1  2=" + std::to_string(sqrt(double(Pips2::latent_dim))) + "\n";
        paramStr += div1Str;
        int h2 = h1 / 2;
        int w2 = w1 / 2;
        std::string reshape2Str = "Reshape /Reshape_2 1 1 /AveragePool_1_output_0_splitncnn_0 /Reshape_2_output_0 2=" + std::to_string(sequenceLength) + " 1=" + std::to_string(Pips2::latent_dim) + " 0=" + std::to_string(h2 * w2) + "\n";
        paramStr += reshape2Str;
        paramStr += "MatMul           /MatMul_2                2 1 feats_splitncnn_1 /Reshape_2_output_0 /MatMul_2_output_0\n";
        std::string div2Str = "BinaryOp  /Div_2  1 1 /MatMul_2_output_0 corrs_pyramid_reshape_2 0=3  1=1  2=" + std::to_string(sqrt(double(Pips2::latent_dim))) + "\n";
        paramStr += div2Str;
        int h3 = h2 / 2;
        int w3 = w2 / 2;
        std::string reshape3Str = "Reshape /Reshape_3 1 1 /AveragePool_2_output_0 /Reshape_3_output_0 2=" + std::to_string(sequenceLength) + " 1=" + std::to_string(Pips2::latent_dim) + " 0=" + std::to_string(h3 * w3) + "\n";
        paramStr += reshape3Str;
        paramStr += "MatMul           /MatMul_3                2 1 feats_splitncnn_0 /Reshape_3_output_0 /MatMul_3_output_0\n";
        std::string div3Str = "BinaryOp  /Div_3  1 1 /MatMul_3_output_0 corrs_pyramid_reshape_3  0=3  1=1  2=" + std::to_string(sqrt(double(Pips2::latent_dim))) + "\n";
        paramStr += div3Str;


        std::string reshape0Str_2 = "Reshape corrs_pyramid_0 1 1 corrs_pyramid_reshape_0 corrs_pyramid_0 2=-1 11=1 1=" + std::to_string(h0) + " 0=" + std::to_string(w0) + "\n";
        paramStr += reshape0Str_2;
        std::string reshape1Str_2 = "Reshape corrs_pyramid_1 1 1 corrs_pyramid_reshape_1 corrs_pyramid_1 2=-1 11=1 1=" + std::to_string(h1) + " 0=" + std::to_string(w1) + "\n";
        paramStr += reshape1Str_2;
        std::string reshape2Str_2 = "Reshape corrs_pyramid_2 1 1 corrs_pyramid_reshape_2 corrs_pyramid_2 2=-1 11=1 1=" + std::to_string(h2) + " 0=" + std::to_string(w2) + "\n";
        paramStr += reshape2Str_2;
        std::string reshape3Str_2 = "Reshape corrs_pyramid_3 1 1 corrs_pyramid_reshape_3 corrs_pyramid_3 2=-1 11=1 1=" + std::to_string(h3) + " 0=" + std::to_string(w3) + "\n";
        paramStr += reshape3Str_2;


        return paramStr;
    }
    std::string Pips2::getDeltaInNer(const int& sequenceLength)
    {
        std::string paramStr = "7767517\n"
            "21 31\n"
            "Input corr1 0 1 corr1\n"///
            "Input corr2 0 1 corr2\n"//
            "Input corr4 0 1 corr4\n"//
            "Input xDiff 0 1 xDiff\n"
            "Input yDiff 0 1 yDiff\n"
            "Input omega 0 1 omega\n"
            "Split splitncnn_xDiff         1 2 xDiff xDiff_0 xDiff_1\n"
            "Split splitncnn_yDiff         1 2 yDiff yDiff_0 yDiff_1\n"
            "Split splitncnn_omega 1 2 omega omega_0 omega_1\n"
            "MatMul omegax 2 1 xDiff_0 omega_0 omegax\n"
            "MatMul omegay 2 1 yDiff_0 omega_1 omegay\n"
            "Split splitncnn_omegax         1 2 omegax omegax_0 omegax_1\n"
            "Split splitncnn_omegay         1 2 omegay omegay_0 omegay_1\n"
            "UnaryOp sin_omegax 1 1 omegax_0 sin_omegax 0=9\n"
            "UnaryOp sin_omegay 1 1 omegay_0 sin_omegay 0=9\n"
            "UnaryOp cos_omegax 1 1 omegax_1 cos_omegax 0=10\n"
            "UnaryOp cos_omegay 1 1 omegay_1 cos_omegay 0=10\n"
            "Concat flow_sincos 6 1 sin_omegax cos_omegax sin_omegay cos_omegay xDiff_1 yDiff_1 flow_sincos 0=1\n"
            "Reshape flow_sincos_reshape 1 1 flow_sincos flow_sincos_reshape 2=-1 1=" + std::to_string(sequenceLength) + " 0=130\n"//
            "Concat deltaIn_permute 4 1 corr1 corr2 corr4 flow_sincos_reshape deltaIn_permute 0=2\n"
            "Permute deltaIn 1 1 deltaIn_permute deltaIn 0=4\n";
            
        return paramStr;
    }
    std::string Pips2::convertDeltaNet(const std::string& paramPath, const int& controlPtsCnt, const int& sequenceLength)
    {
        std::string paramString = "";
        std::fstream fin(paramPath, std::ios::in);
        std::string aline;
        std::list<std::string>paramLines;
        int additionalCnt = 0;
        while (std::getline(fin, aline))
        {
            std::vector<std::string> segs = splitString(aline, " ", true);
            if (segs[0].compare("InstanceNorm") == 0)
            {
                additionalCnt += 2;
                std::string& layerName = segs[1];
                std::string& inPutName = segs[4];
                std::string& outPutName = segs[5];
                std::string reshape1Str = "Reshape " + (layerName + "_reshape1") + " 1 1 " + inPutName + " " + (inPutName + "_reshape") + " 2=-1 1=1 0=" + std::to_string(sequenceLength);
                paramLines.emplace_back(reshape1Str);
                inPutName = inPutName + "_reshape";
                std::string outPutNameOld = outPutName;
                outPutName = outPutName + "_reshape";
                std::string InstanceNormLine = "";
                for (int i = 0; i < segs.size(); i++)
                {
                    InstanceNormLine += segs[i];
                    InstanceNormLine += " ";
                }
                paramLines.emplace_back(InstanceNormLine);
                std::string reshape2Str = "Reshape " + (layerName + "_reshape2") + " 1 1 " + outPutName + " " + outPutNameOld + " 2=-1 1=" + std::to_string(controlPtsCnt) + " 0=" + std::to_string(sequenceLength);
                paramLines.emplace_back(reshape2Str);
            }
            else
            {
                paramLines.emplace_back(aline);
            }
        }
        int lineId = 0;
        for (const auto& d : paramLines)
        {
            if (lineId == 1)
            {
                std::stringstream ss;
                ss << d;
                int a, b;
                ss >> a >> b;
                paramString += (std::to_string(a + additionalCnt) + " " + std::to_string(b + additionalCnt));
                paramString += '\n';
            }
            else
            {
                paramString += d;
                paramString += '\n';
            }
            lineId++;
        }
        //std::cout << paramString << std::endl;
        return paramString;
    }
    ncnn::Mat Pips2::bilinear_sample2d(const ncnn::Mat& blob, const std::vector<float>& xs, const std::vector<float>& ys, std::shared_ptr<ncnn::Net> bilinearOpNet)
    {
        int ptsCnt = xs.size();
        int featDim = 0;
        float W_f = blob.w;
        float H_f = blob.h;
        int max_x = blob.w - 1;
        int max_y = blob.h - 1;
        ncnn::Mat value;
        ncnn::Mat weight;
        int outC = 0;
        int outH = 0;
        int outW = 0;
        if (blob.dims==3)
        {
            featDim = blob.c;
            value = ncnn::Mat(4, featDim, ptsCnt, (size_t)4);
            weight = ncnn::Mat(1, 4, ptsCnt, (size_t)4);
            outC = 1;
            outH = ptsCnt;
            outW = featDim;
        }
        else if (blob.dims == 4)
        {
            featDim = blob.d;
            value = ncnn::Mat(4, featDim, ptsCnt * blob.c, (size_t)4);
            weight = ncnn::Mat(1, 4, ptsCnt * blob.c, (size_t)4);
            outC = blob.c;
            outH = ptsCnt;
            outW = featDim;
        }
        else
        {
            LOG_ERR_OUT << "dims error";
            return ncnn::Mat();
        }
        int value_cstep = value.cstep;
        int weight_cstep = weight.cstep;
        int C = weight.c;
        for (size_t i = 0; i < ptsCnt; i++)
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
            float w_y0_x0 = (x1 - xs[i]) * (y1 - ys[i]);
            float w_y0_x1 = (xs[i] - x0) * (y1 - ys[i]);
            float w_y1_x0 = (x1 - xs[i]) * (ys[i] - y0);
            float w_y1_x1 = (xs[i] - x0) * (ys[i] - y0);
            int p0 = x0 + blob.w * y0;
            int p1 = x1 + blob.w * y0;
            int p2 = x0 + blob.w * y1;
            int p3 = x1 + blob.w * y1;
            if (blob.dims == 3)
            {
                int pp = i * weight_cstep;
                ((float*)weight.data)[pp] = w_y0_x0;
                ((float*)weight.data)[pp + 1] = w_y0_x1;
                ((float*)weight.data)[pp + 2] = w_y1_x0;
                ((float*)weight.data)[pp + 3] = w_y1_x1;
                {
                    int pp0 = i * value_cstep;
                    for (int fea = 0; fea < featDim; fea++)
                    {
                        ((float*)value.data)[pp0 + fea * 4] = ((const float*)blob.data)[p0 + fea * blob.cstep];
                        ((float*)value.data)[pp0 + fea * 4 + 1] = ((const float*)blob.data)[p1 + fea * blob.cstep];
                        ((float*)value.data)[pp0 + fea * 4 + 2] = ((const float*)blob.data)[p2 + fea * blob.cstep];
                        ((float*)value.data)[pp0 + fea * 4 + 3] = ((const float*)blob.data)[p3 + fea * blob.cstep];
                    }
                }
            }
            else if (blob.dims == 4)
            {
                int whStep = blob.w * blob.h;
                for (int c = 0; c < blob.c; c++)
                {
                    int pp = (i + c * ptsCnt) * weight_cstep;// +i * 4;
                    ((float*)weight.data)[pp] = w_y0_x0;
                    ((float*)weight.data)[pp + 1] = w_y0_x1;
                    ((float*)weight.data)[pp + 2] = w_y1_x0;
                    ((float*)weight.data)[pp + 3] = w_y1_x1;
                    int pp0 = (i + c * ptsCnt) * value_cstep;
                    for (int fea = 0; fea < featDim; fea++)
                    {
                        int j = whStep * fea;
                        ((float*)value.data)[pp0 + fea * 4 + 0] = ((const float*)blob.data)[p0 + j + c * blob.cstep];
                        ((float*)value.data)[pp0 + fea * 4 + 1] = ((const float*)blob.data)[p1 + j + c * blob.cstep];
                        ((float*)value.data)[pp0 + fea * 4 + 2] = ((const float*)blob.data)[p2 + j + c * blob.cstep];
                        ((float*)value.data)[pp0 + fea * 4 + 3] = ((const float*)blob.data)[p3 + j + c * blob.cstep];
                    }
                }
            } 
        }
        ncnn::Extractor ex2 = bilinearOpNet->create_extractor();
        ex2.input("v", value);
        ex2.input("w", weight);
        auto start1 = std::chrono::steady_clock::now();
        ncnn::Mat bilinear_sample_out;
        ex2.extract("output", bilinear_sample_out);
        auto end1 = std::chrono::steady_clock::now();
        auto elapsed1 = std::chrono::duration_cast<std::chrono::milliseconds>(end1 - start1).count();
        //dnn::ncnnHelper::printBlob(weight);
        std::cout << "Elapsed time: " << elapsed1 << " ms" << std::endl;
        return bilinear_sample_out.reshape(outW, outH, outC);;
    }

    ncnn::Mat Pips2::bilinear_sample2d(const ncnn::Mat& blob, const std::vector<std::vector<float>>& xs, const std::vector<std::vector<float>>& ys, std::shared_ptr<ncnn::Net> bilinearOpNet, const int&padding_mode)
    {
        int ptsCnt = xs[0].size();
        int featDim = 0;
        float W_f = blob.w;
        float H_f = blob.h;
        int max_x = blob.w - 1;
        int max_y = blob.h - 1;
        int max_x_1 = max_x - 1;
        int max_y_1 = max_y - 1;
        ncnn::Mat value;
        ncnn::Mat weight;
        int outC = 0;
        int outH = 0;
        int outW = 0;
        if (blob.dims == 4)
        {
            featDim = blob.d;
            value = ncnn::Mat(4, featDim, ptsCnt * blob.c, (size_t)4);
            weight = ncnn::Mat(1, 4, ptsCnt * blob.c, (size_t)4);
            outC = blob.c;
            outH = ptsCnt;
            outW = featDim;
        }
        else
        {
            //std::vector<float>xsFlat;
            //std::vector<float>ysFlat;
            //xsFlat.reserve(xs.size() * xs[0].size());
            //ysFlat.reserve(ys.size() * ys[0].size());
            //for (int i = 0; i < xs.size(); i++)
            //{
            //    for (int j = 0; j < xs[i].size(); j++)
            //    {
            //        xsFlat.emplace_back(xs[i][j]);
            //        ysFlat.emplace_back(ys[i][j]);
            //    }
            //}
            //ncnn::Mat sampledFlat = bilinear_sample2d(blob, xsFlat, ysFlat, bilinearOpNet);
            //return sampledFlat;
            LOG_ERR_OUT << "dims error";
            return ncnn::Mat();
        }
        if (blob.c!= xs.size())
        { 
            LOG_ERR_OUT << "size error";
            return ncnn::Mat();
        }
        int value_cstep = value.cstep;
        int weight_cstep = weight.cstep;
        int C = weight.c;

        for (int c = 0; c < blob.c; c++)
        {
            for (size_t i = 0; i < ptsCnt; i++)
            {
                int x0 = std::floor(xs[c][i]);
                int x1 = x0 + 1;
                int y0 = std::floor(ys[c][i]);
                int y1 = y0 + 1;
                float w_y0_x0 = (x1 - xs[c][i]) * (y1 - ys[c][i]);
                float w_y0_x1 = (xs[c][i] - x0) * (y1 - ys[c][i]);
                float w_y1_x0 = (x1 - xs[c][i]) * (ys[c][i] - y0);
                float w_y1_x1 = (xs[c][i] - x0) * (ys[c][i] - y0);

                if (x0 < 0 || x0 > max_x) { x0 = 0; w_y0_x0 = 0; w_y1_x0 = 0; }
                if (y0 < 0 || y0 > max_y) { y0 = 0; w_y0_x0 = 0; w_y0_x1 = 0; }
                if (x1 < 0 || x1 > max_x) { x1 = 0; w_y0_x1 = 0; w_y1_x1 = 0; }
                if (y1 < 0 || y1 > max_y) { y1 = 0; w_y1_x0 = 0; w_y1_x1 = 0; }


                int p0 = x0 + blob.w * y0;
                int p1 = x1 + blob.w * y0;
                int p2 = x0 + blob.w * y1;
                int p3 = x1 + blob.w * y1;
                int whStep = blob.w * blob.h;                
                int pp = (i + c * ptsCnt) * weight_cstep;// +i * 4;
                ((float*)weight.data)[pp] = w_y0_x0;
                ((float*)weight.data)[pp + 1] = w_y0_x1;
                ((float*)weight.data)[pp + 2] = w_y1_x0;
                ((float*)weight.data)[pp + 3] = w_y1_x1;
                int pp0 = (i + c * ptsCnt) * value_cstep;
                for (int fea = 0; fea < featDim; fea++)
                {
                    int j = whStep * fea;
                    ((float*)value.data)[pp0 + fea * 4 + 0] = ((const float*)blob.data)[p0 + j + c * blob.cstep];
                    ((float*)value.data)[pp0 + fea * 4 + 1] = ((const float*)blob.data)[p1 + j + c * blob.cstep];
                    ((float*)value.data)[pp0 + fea * 4 + 2] = ((const float*)blob.data)[p2 + j + c * blob.cstep];
                    ((float*)value.data)[pp0 + fea * 4 + 3] = ((const float*)blob.data)[p3 + j + c * blob.cstep];
                }
            }
        }
        ncnn::Extractor ex2 = bilinearOpNet->create_extractor();
        ex2.input("v", value);
        ex2.input("w", weight);
        auto start1 = std::chrono::steady_clock::now();
        ncnn::Mat bilinear_sample_out;
        ex2.extract("output", bilinear_sample_out);
        auto end1 = std::chrono::steady_clock::now();
        auto elapsed1 = std::chrono::duration_cast<std::chrono::milliseconds>(end1 - start1).count();
        //dnn::ncnnHelper::printBlob(value);
        //dnn::ncnnHelper::printBlob(weight);
        std::cout << "Elapsed time: " << elapsed1 << " ms" << std::endl;
        return bilinear_sample_out.reshape(outW, outH, outC);;
    }
    ncnn::Mat Pips2::concatFmaps(const std::vector<ncnn::Mat>&fmap, const std::vector<int>& picks)
    {
        int w = fmap[0].w;
        int h = fmap[0].h;
        int cstep = fmap[0].cstep;
        ncnn::Mat fmaps(w,h, (int)(Pips2::latent_dim * picks.size()), (size_t)4);
        for (int i = 0; i < picks.size(); i++)
        {
            memcpy((float*)fmaps.data + i * cstep * 128, fmap[picks[i]].data, cstep * Pips2::latent_dim * sizeof(float));
        }
        return fmaps;
    }
    ncnn::Mat Pips2::concatFmapsWithBatch(const std::vector<ncnn::Mat>& fmap, const std::vector<int>& picks)
    {
        if (fmap[0].dims!=3)
        {
            LOG_ERR_OUT << "dims must be 3";
            return ncnn::Mat();
        }
        int w = fmap[0].w;
        int h = fmap[0].h;
        int cstep = fmap[0].cstep;
        ncnn::Mat fmaps(w, h, fmap[0].c, (int)fmap.size(),(size_t)4);
        for (int i = 0; i < picks.size(); i++)
        {
            float* target_ = (float*)fmaps.data + i * fmaps.cstep;
            int innerCnt = fmap[i].h * fmap[i].w;
            for (int c = 0; c < fmap[i].c; c++)
            {
                float* target = target_ + c * innerCnt;
                memcpy(target, (float*)fmap[picks[i]].data+c* fmap[i].cstep, innerCnt * sizeof(float));
            }
        }
        return fmaps;
    }
    ncnn::Mat Pips2::repeatFeat(const ncnn::Mat&feat, const int& s)
    {
        int w = feat.w;
        int h = feat.h;
        int c = feat.c;
        int cstep = feat.cstep;
        ncnn::Mat feats(w, h, (int)(c * s), (size_t)4);
        for (int i = 0; i < s; i++)
        {
            memcpy((float*)feats.data + i * cstep * c, feat.data, cstep * c * sizeof(float));
        }
        return feats;
    }
    std::vector<std::vector<float>> Pips2::expandInitCoord(std::vector<float>& xs, const int& times)
    {
        std::vector<std::vector<float>>xs_(times);
        for (size_t i = 0; i < times; i++)
        {
            xs_[i].insert(xs_[i].end(), xs.begin(), xs.end());
        } 
        return xs_;
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
        const std::filesystem::path& ncnnEncoderParamPath,
        const std::filesystem::path& ncnnEncoderBinPath,
        const std::filesystem::path& ncnnDeltaBlockParamPath,
        const std::filesystem::path& ncnnDeltaBlockBinPath,
        const int& radius_)
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
        if (!std::filesystem::exists(ncnnDeltaBlockParamPath))
        {
            LOG_ERR_OUT << "not found : " << ncnnDeltaBlockParamPath;
            return;
        }
        if (!std::filesystem::exists(ncnnDeltaBlockBinPath))
        {
            LOG_ERR_OUT << "not found : " << ncnnDeltaBlockBinPath;
            return;
        }
        ncnnEncoderParamPath_ = ncnnEncoderParamPath;
        ncnnEncoderBinPath_ = ncnnEncoderBinPath;
        ncnnDeltaBlockParamPath_ = ncnnDeltaBlockParamPath;
        ncnnDeltaBlockBinPath_ = ncnnDeltaBlockBinPath;

        std::string paramStr = pips2::Pips2::getBilinearOpNet();
        bilinearOpNet = std::shared_ptr<ncnn::Net>(new ncnn::Net());
        bilinearOpNet->load_param_mem(paramStr.c_str());
        bilinearOpNet->load_model((const unsigned char*)0);
        radius = radius_;

        coord_delta_x.clear();
        coord_delta_y.clear();
        coord_delta_x.reserve(4 * radius * radius + 4 * radius + 1);
        coord_delta_y.reserve(4 * radius * radius + 4 * radius + 1);
        for (int c = -radius; c <= radius; c++)
        {
            for (int r = -radius; r <= radius; r++)
            {
                coord_delta_x.emplace_back(c);
                coord_delta_y.emplace_back(r);
            }
        }


        omega = ncnn::Mat(Pips2::latent_dim/4, 1, (size_t)4);
        double omegaEachLinear = (1. / omega.w-1);
        for (int i = 0; i < omega.w; i++)
        {
            if (i== omega.w-1)
            {
                ((float*)omega.data)[i] = 1. / Pips2::omega_temperature;
            }
            else
            {
                ((float*)omega.data)[i] = 1. / std::pow(Pips2::omega_temperature, static_cast<double>(i) / (omega.w - 1));
            }
        }
        return;
    }
    Pips2::~Pips2()
    {}
    bool Pips2::initDeltaBlockNet(const int& controlPtsCnt, const int& sequenceCnt)
    {
        try
        {
            deltaNet = std::shared_ptr<ncnn::Net>(new ncnn::Net());
            std::string deltaNetParamStr = convertDeltaNet(ncnnDeltaBlockParamPath_.string(), controlPtsCnt, sequenceCnt);
            deltaNet->load_param_mem(deltaNetParamStr.c_str());
            deltaNet->load_model(ncnnDeltaBlockBinPath_.string().c_str());
            padding64data = std::vector<float>(controlPtsCnt * sequenceCnt * 64);
            padding128data = std::vector<float>(controlPtsCnt * sequenceCnt * 128);
            padding256data = std::vector<float>(controlPtsCnt * sequenceCnt * 256);
            padding64 = ncnn::Mat(sequenceCnt, controlPtsCnt, 64, (void*)&padding256data[0], 4);
            padding128 = ncnn::Mat(sequenceCnt, controlPtsCnt, 128, (void*)&padding256data[0], 4);
            padding256 = ncnn::Mat(sequenceCnt, controlPtsCnt, 256, (void*)&padding256data[0], 4);
            padding64b = ncnn::Mat(sequenceCnt, controlPtsCnt, 64, (void*)&padding256data[0], 4);
            padding128b = ncnn::Mat(sequenceCnt, controlPtsCnt, 128, (void*)&padding256data[0], 4);
            padding256b = ncnn::Mat(sequenceCnt, controlPtsCnt, 256, (void*)&padding256data[0], 4);


        }
        catch (const std::exception&)
        {
            LOG_ERR_OUT << "init DeltaBlockNet error!";
            return false;
        }
        return true;
    }
    bool Pips2::inputImage(const std::vector<std::string>& imgPath,std::vector<ncnn::Mat>&fmaps)
    {
        if (imgPath.size()==0)
        {
            LOG_ERR_OUT << "imgPath.size()==0";
            return false;
        }
        fmaps.clear();
        fmaps.reserve(imgPath.size());

        positionDiffEncoderNet = std::shared_ptr<ncnn::Net>(new ncnn::Net());
        positionDiffEncoderNet->load_param_mem(pips2::Pips2::getDeltaInNer(imgPath.size()).c_str());
        positionDiffEncoderNet->load_model((const unsigned char*)0);



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
                fmapPyramidSize.clear();
                fmapPyramidSize.resize(pyramid_level);
                for (size_t i = 0; i < Pips2::pyramid_level; i++)
                {
                    if (i==0)
                    {
                        fmapPyramidSize[i].width = imgSize.width / Pips2::stride;
                        fmapPyramidSize[i].height = imgSize.height / Pips2::stride;
                    }
                    else
                    {
                        fmapPyramidSize[i].width = fmapPyramidSize[i - 1].width / 2;
                        fmapPyramidSize[i].height = fmapPyramidSize[i - 1].height / 2;
                    }
                }
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


                std::string paramStr = pips2::Pips2::getCorrsNet(imgPath.size(), imgSize.height, imgSize.width);
                corrsNet = std::shared_ptr<ncnn::Net>(new ncnn::Net());
                corrsNet->load_param_mem(paramStr.c_str());
                corrsNet->load_model((const unsigned char*)0);

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

    ncnn::Mat Pips2::pyramidSample(const std::vector<ncnn::Mat>& corrs_pyramids, const std::vector<std::vector<float>>& stride_x, const std::vector<std::vector<float>>& stride_y)const
    {
        int radiusArea = coord_delta_x.size();
        int imgSeqCnt = stride_x.size(); 
        int controlPtsCnt = stride_x[0].size();
        ncnn::Mat corrOut(radiusArea * Pips2::pyramid_level, imgSeqCnt, controlPtsCnt, (size_t)4);

        
        for (size_t i = 0; i < Pips2::pyramid_level; i++)
        {
            std::vector<std::vector<float>>xs_(stride_x.size() * stride_x[0].size());
            std::vector<std::vector<float>>ys_(stride_y.size() * stride_y[0].size());
            for (int c = 0; c < stride_x.size() * stride_x[0].size(); c++)
            {
                int m = c / stride_x[0].size();
                int n = c % stride_x[0].size();
                xs_[c].resize(coord_delta_x.size());
                ys_[c].resize(coord_delta_y.size());
                {
                    float x = stride_x[m][n] / std::pow(2, i);
                    float y = stride_y[m][n] / std::pow(2, i);
                    for (size_t j = 0; j < coord_delta_x.size(); j++)
                    {
                        xs_[c][j] = coord_delta_x[j] + x;
                        ys_[c][j] = coord_delta_y[j] + y;
                    }
                }
            }
            ncnn::Mat corr = bilinear_sample2d(corrs_pyramids[i], xs_, ys_, bilinearOpNet);
            //using dnn::ocvHelper::operator<<;
            //using dnn::ncnnHelper::operator<<;
            //LOG_OUT << corr;
            for (size_t c = 0; c < corr.c; c++)
            {
                int seqIdx = c / 3;
                int controlPtIdx = c % 3;
                float* tar = (float*)corrOut.data + controlPtIdx * corrOut.cstep + seqIdx * corrOut.w + i * radiusArea;
                float* src = (float*)corr.data + c * corr.cstep;
                memcpy(tar, src, radiusArea * sizeof(float));
            }
        }
        return corrOut;
    }
    ncnn::Mat Pips2::fillPositionDiffCosSin(const ncnn::Mat& corr1, const ncnn::Mat& corr2, const ncnn::Mat& corr4, const std::vector<std::vector<float>>& stride_x, const std::vector<std::vector<float>>& stride_y)
    {
        int positionCnt = stride_x.size() * stride_x[0].size();
        int controlPtsCnt = stride_x[0].size();
        int seqenceCnt = stride_x.size();
        ncnn::Mat diffx(1, positionCnt, (size_t)4);
        ncnn::Mat diffy(1, positionCnt, (size_t)4);
        for (int i = 0; i < positionCnt; i++)
        {
            int controlPtIdx = i % controlPtsCnt;
            int seqenceIdx = i / controlPtsCnt;
            int seqenceNextIdx = seqenceIdx+1;
            if (seqenceNextIdx == seqenceCnt)
            {
                ((float*)diffx.data)[i] = stride_x[seqenceIdx][controlPtIdx] - stride_x[seqenceIdx - 1][controlPtIdx];
                ((float*)diffy.data)[i] = stride_y[seqenceIdx][controlPtIdx] - stride_y[seqenceIdx - 1][controlPtIdx];
            }
            else
            {
                ((float*)diffx.data)[i] = stride_x[seqenceNextIdx][controlPtIdx] - stride_x[seqenceIdx][controlPtIdx];
                ((float*)diffy.data)[i] = stride_y[seqenceNextIdx][controlPtIdx] - stride_y[seqenceIdx][controlPtIdx];
            }
        }
        ncnn::Extractor ex2 = positionDiffEncoderNet->create_extractor();
        ex2.input("corr1", corr1);
        ex2.input("corr2", corr2);
        ex2.input("corr4", corr4);
        ex2.input("xDiff", diffx);
        ex2.input("yDiff", diffy);
        ex2.input("omega", omega);
        auto start1 = std::chrono::steady_clock::now();
        ncnn::Mat deltaIn;
        ex2.extract("deltaIn", deltaIn);
        auto end1 = std::chrono::steady_clock::now();
        auto elapsed1 = std::chrono::duration_cast<std::chrono::milliseconds>(end1 - start1).count();
        //dnn::ncnnHelper::printBlob(deltaIn);
        //dnn::ncnnHelper::printBlob(deltaIn);
        std::cout << "Elapsed time: " << elapsed1 << " ms" << std::endl; 
        return deltaIn;
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
    using dnn::ocvHelper::operator<<;
    using dnn::ncnnHelper::operator<<;
    std::string paramStr = pips2::Pips2::getBilinearOpNet();
    std::shared_ptr<ncnn::Net> bilinearOpNet(new ncnn::Net());
    bilinearOpNet->load_param_mem(paramStr.c_str());
    bilinearOpNet->load_model((const unsigned char*)0);

    std::vector<int>shape = {8, 128,64,64 }; ;// {8, 128, 64, 64};
    int totalcnt = std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<int>());
    std::vector<float> indata(totalcnt);
    for (int i = 0; i < indata.size(); i++)
    {
        indata[i] = i%200-100;
    }
    ncnn::Mat in;
    if (shape.size()==3)
    {
        in = ncnn::Mat(shape[2], shape[1], shape[0], (void*)&indata[0], 4);
    }
    else if(shape.size() == 4)
    {
        in = ncnn::Mat(shape[3], shape[2], shape[1], shape[0], (void*)&indata[0], 4);
    }
    std::vector<float>xs = { 12.5000, 13.3000, 23.3000 };
    std::vector<float>ys = { 1.2000, 45.1000, 15.1000 };

    ncnn::Mat bilinearOut = pips2::Pips2::bilinear_sample2d(in, xs, ys, bilinearOpNet);
    LOG_OUT << bilinearOut;
    return 0;
}
int test_pips2()
{
    //return test_deltaNet();
    //return test_bilinearOp();
    using dnn::ocvHelper::operator<<;
    using dnn::ncnnHelper::operator<<;

    std::vector<std::string>paths = {
    "D:/repo/colmapThird/data2/a/00000.jpg", 
    "D:/repo/colmapThird/data2/a/00001.jpg", 
    "D:/repo/colmapThird/data2/a/00002.jpg", 
    "D:/repo/colmapThird/data2/a/00003.jpg", 
    "D:/repo/colmapThird/data2/a/00004.jpg", 
    "D:/repo/colmapThird/data2/a/00005.jpg", 
    "D:/repo/colmapThird/data2/a/00006.jpg", 
    "D:/repo/colmapThird/data2/a/00007.jpg", };

    pips2::Pips2 ins("../models/pips2_base_ncnn.param", "../models/pips2_base_ncnn.bin", "../models/pips2_deltaBlock_ncnn.param", "../models/pips2_deltaBlock_ncnn.bin");
    std::vector<ncnn::Mat> fmapsVec;
    ins.inputImage(paths, fmapsVec);
    LOG_OUT<< fmapsVec[0];
    std::vector<float>xs0 = { 12.5000, 13.3000, 23.3000 };
    std::vector<float>ys0 = { 1.2000, 45.1000, 15.1000 };
    for (auto& d : xs0) d /= ins.stride;
    for (auto& d : ys0) d /= ins.stride;
    ncnn::Mat feat0 = ins.bilinear_sample2d(fmapsVec[0], xs0, ys0, ins.bilinearOpNet);
    LOG_OUT << feat0; 

    std::vector<int>initFrameIdx = { 0,1,2,3,4,5,6,7 };
    ncnn::Mat fmaps = pips2::Pips2::concatFmaps(fmapsVec, initFrameIdx);
    ins.initDeltaBlockNet(xs0.size(), fmapsVec.size());
    ncnn::Mat feats = pips2::Pips2::repeatFeat(feat0, paths.size());
    std::vector<std::vector<float>>xs = pips2::Pips2::expandInitCoord(xs0, paths.size());
    std::vector<std::vector<float>>ys = pips2::Pips2::expandInitCoord(ys0, paths.size());

    ncnn::Extractor ex_corrs1 = ins.corrsNet->create_extractor();
    ex_corrs1.input("fmaps", fmaps);
    ex_corrs1.input("feats", feats);
    ncnn::Mat corrs1_pyramid0, corrs1_pyramid1, corrs1_pyramid2, corrs1_pyramid3;
    ex_corrs1.extract("corrs_pyramid_0", corrs1_pyramid0);
    ex_corrs1.extract("corrs_pyramid_1", corrs1_pyramid1);
    ex_corrs1.extract("corrs_pyramid_2", corrs1_pyramid2);
    ex_corrs1.extract("corrs_pyramid_3", corrs1_pyramid3);
    ncnn::Mat corrs2_pyramid0, corrs2_pyramid1, corrs2_pyramid2, corrs2_pyramid3;
    ncnn::Mat corrs4_pyramid0, corrs4_pyramid1, corrs4_pyramid2, corrs4_pyramid3;
    corrs2_pyramid0.clone_from(corrs1_pyramid0);
    corrs4_pyramid0.clone_from(corrs1_pyramid0);
    corrs2_pyramid1.clone_from(corrs1_pyramid1);
    corrs4_pyramid1.clone_from(corrs1_pyramid1);
    corrs2_pyramid2.clone_from(corrs1_pyramid2);
    corrs4_pyramid2.clone_from(corrs1_pyramid2);
    corrs2_pyramid3.clone_from(corrs1_pyramid3);
    corrs4_pyramid3.clone_from(corrs1_pyramid3);

   

    for (size_t iter = 0; iter < 3; iter++)
    {
        if (iter>=1)
        {
            int inds2 = 2;
            int inds4 = 4;
            std::vector<int>frameIdx2(fmapsVec.size()), frameIdx4(fmapsVec.size());
            std::vector<std::vector<float>>xs2(xs.size(), std::vector<float>(xs[0].size(), 0));
            std::vector<std::vector<float>>ys2(ys.size(), std::vector<float>(ys[0].size(), 0));
            std::vector<std::vector<float>>xs4(xs.size(), std::vector<float>(xs[0].size(), 0));
            std::vector<std::vector<float>>ys4(ys.size(), std::vector<float>(ys[0].size(), 0));
            for (size_t f = 0; f < fmapsVec.size(); f++)
            {
                frameIdx2[f] = initFrameIdx[f] - inds2;
                frameIdx4[f] = initFrameIdx[f] - inds4;
                if (frameIdx2[f] < 0)frameIdx2[f] = 0;
                if (frameIdx4[f] < 0)frameIdx4[f] = 0;
                memcpy(&xs2[f][0], &xs[frameIdx2[f]][0], sizeof(float) * xs[0].size());
                memcpy(&ys2[f][0], &ys[frameIdx2[f]][0], sizeof(float) * ys[0].size());
                memcpy(&xs4[f][0], &xs[frameIdx4[f]][0], sizeof(float) * xs[0].size());
                memcpy(&ys4[f][0], &ys[frameIdx4[f]][0], sizeof(float) * ys[0].size());
            }
            ncnn::Mat fmaps2 = pips2::Pips2::concatFmapsWithBatch(fmapsVec, frameIdx2);
            ncnn::Mat fmaps4 = pips2::Pips2::concatFmapsWithBatch(fmapsVec, frameIdx4);


            ncnn::Mat feats2 = ins.bilinear_sample2d(fmaps2, xs2, ys2, ins.bilinearOpNet);
            ncnn::Mat feats4 = ins.bilinear_sample2d(fmaps2, xs2, ys2, ins.bilinearOpNet);


            ncnn::Extractor ex_corrs2 = ins.corrsNet->create_extractor();
            ex_corrs2.input("fmaps", fmaps2);
            ex_corrs2.input("feats", feats2);
            ex_corrs2.extract("corrs_pyramid_0", corrs2_pyramid0);
            ex_corrs2.extract("corrs_pyramid_1", corrs2_pyramid1);
            ex_corrs2.extract("corrs_pyramid_2", corrs2_pyramid2);
            ex_corrs2.extract("corrs_pyramid_3", corrs2_pyramid3);
            ncnn::Extractor ex_corrs4 = ins.corrsNet->create_extractor();
            ex_corrs4.input("fmaps", fmaps4);
            ex_corrs4.input("feats", feats4);
            ex_corrs4.extract("corrs_pyramid_0", corrs4_pyramid0);
            ex_corrs4.extract("corrs_pyramid_1", corrs4_pyramid1);
            ex_corrs4.extract("corrs_pyramid_2", corrs4_pyramid2);
            ex_corrs4.extract("corrs_pyramid_3", corrs4_pyramid3);
        }
        ncnn::Mat corrs1 = ins.pyramidSample({ corrs1_pyramid0, corrs1_pyramid1, corrs1_pyramid2, corrs1_pyramid3 }, xs, ys);
        ncnn::Mat corrs2 = ins.pyramidSample({ corrs2_pyramid0, corrs2_pyramid1, corrs2_pyramid2, corrs2_pyramid3 }, xs, ys);
        ncnn::Mat corrs4 = ins.pyramidSample({ corrs4_pyramid0, corrs4_pyramid1, corrs4_pyramid2, corrs4_pyramid3 }, xs, ys);
        
        if (dnn::ncnnHelper::dataHasNanInf(corrs1))
        {
            LOG_ERR_OUT << "nan in corrs1";
            return -1;
        }
        if (dnn::ncnnHelper::dataHasNanInf(corrs2))
        {
            LOG_ERR_OUT << "nan in corrs2";
            return -1;
        }
        if (dnn::ncnnHelper::dataHasNanInf(corrs4))
        {
            LOG_ERR_OUT << "nan in corrs4";
            return -1;
        }


        ncnn::Mat deltaNetInput = ins.fillPositionDiffCosSin(corrs1, corrs2, corrs4, xs, ys);



        //dnn::ncnnHelper::printBlob(deltaNetInput);

        ncnn::Extractor ex3 = ins.deltaNet->create_extractor();
        ex3.input("deltaIn", deltaNetInput);
        ex3.input("padding64", ins.padding64);
        ex3.input("padding128", ins.padding128);
        ex3.input("padding256", ins.padding256);
        ex3.input("padding64b", ins.padding64);
        ex3.input("padding128b", ins.padding128);
        ex3.input("padding256b", ins.padding256);
        ncnn::Mat delta_out;
        ex3.extract("delta", delta_out);
        //dnn::ncnnHelper::printBlob(delta_out);
        
        if (iter==1)
        {
            LOG_OUT;
        }

        int blobDataI = 0;
        for (int p = 0; p < xs[0].size(); p++)
        {
            for (int q = 0; q < xs.size(); q++)
            {
                xs[q][p] += ((float*)delta_out.data)[blobDataI++];
                ys[q][p] += ((float*)delta_out.data)[blobDataI++];
            }
        }
        xs[0] = xs0;
        ys[0] = ys0;
        LOG_OUT;
    }

    return 0;
}