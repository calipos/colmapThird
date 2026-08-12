#include "pips2iter.h"

int test_pips2_iter()
{
    iter::Pips2<2,12> ins("../models/pips2_base_ncnn.param", "../models/pips2_base_ncnn.bin", "../models/pips2_deltaBlock_ncnn.param", "../models/pips2_deltaBlock_ncnn.bin", 1280, 720);

    cv::Mat img0 = cv::imread("../data/a/00000.jpg");
    ncnn::Mat fmap0;
    ins.extractFeat(img0, fmap0);

    cv::Mat img1 = cv::imread("../data/a/00001.jpg");
    ncnn::Mat fmap1;
    ins.extractFeat(img1, fmap1);



    std::vector<cv::Point2f>  controlPts = { {362,565} };
    std::array<ncnn::Mat, 2> feats{ fmap0 ,fmap1 };
    std::array<std::vector<cv::Point2f>, 2> traj;
    ins.track(controlPts, feats, traj);
    LOG_OUT << traj[0][0];
    LOG_OUT << traj[1][0];

    return 0;
}


 