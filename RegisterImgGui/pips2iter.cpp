#include "pips2iter.h"
#include "dnnHelper.h"
int test_pips2_iter()
{
    using dnn::ocvHelper::operator<<; 
    using dnn::ncnnHelper::operator<<;
    iter::Pips2<6,12> ins("../models/pips2_base_ncnn.param", "../models/pips2_base_ncnn.bin", "../models/pips2_deltaBlock_ncnn.param", "../models/pips2_deltaBlock_ncnn.bin", 1280, 720);

    cv::Mat img0 = cv::imread("../data/c/00000.jpg");
    ncnn::Mat fmap0;
    ins.extractFeat(img0, fmap0);

    cv::Mat img1 = cv::imread("../data/c/00001.jpg");
    ncnn::Mat fmap1;
    ins.extractFeat(img1, fmap1);

    ncnn::Mat fmap2;
    ncnn::Mat fmap3;
    ncnn::Mat fmap4;
    ncnn::Mat fmap5;

    fmap2.clone_from(fmap1);
    fmap3.clone_from(fmap1);
    fmap4.clone_from(fmap1);
    fmap5.clone_from(fmap1);
 

    std::vector<cv::Point2f>  controlPts = { {362,565} };
    std::vector<ncnn::Mat> feats{ fmap0 ,fmap1,         
        fmap2,fmap3,fmap4,fmap5,  
    };
    std::vector<std::vector<cv::Point2f>> traj;
    ins.track(controlPts, feats, traj);
    LOG_OUT << traj[0][0];
    LOG_OUT << traj[1][0];

    return 0;
}


 