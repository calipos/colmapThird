#include "pips2.h"
#include "opencv2/opencv.hpp"
#include <fstream>
#include <numeric>
#include <iostream>
#include <map>
#include <unordered_set>
#include <chrono>
#include <vector>
#include "opencv2/dnn/dnn.hpp"
#include "opencv2/dnn/layer.hpp"
#include "opencv2/dnn/shape_utils.hpp"
int test_pips2()
{ 
    cv::dnn::Net Net = cv::dnn::readNetFromONNX("../models/pips2_base_opencv.onnx");
    std::cout << "load EncoderNet." << std::endl;
    int sz[] = { 8,3,1024,1024 };

    //Net.setPreferableBackend(cv::dnn::DNN_TARGET_CPU);
    //Net.setInput(imgBlob);
    //std::vector<cv::Mat> imgEncoderNetOut;
    //std::vector<std::string> outLayersNames = { "high_res_feats_0","high_res_feats_1","image_embed" };
    //auto start1 = std::chrono::steady_clock::now();
    //imgEncoderNet.forward(imgEncoderNetOut, outLayersNames);  // crash here
    //auto end1 = std::chrono::steady_clock::now();
    //auto elapsed1 = std::chrono::duration_cast<std::chrono::milliseconds>(end1 - start1).count();
    //std::cout << "Elapsed time: " << elapsed1 << " ms" << std::endl;
    //std::cout << "encode forward ok " << std::endl;
	return 0;
}