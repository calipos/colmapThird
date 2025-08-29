#include "log.h"
#include "net.h"
#include "opencv2/opencv.hpp"
#include <string>
#include <optional>
#include <filesystem>
#include <numeric>
#include <fstream>
#include "sam2.h"
#include "dnnHelper.h"

namespace sam2
{
    //"/Reshape_12_output_0", "/GreaterOrEqual_output_0","/iou_prediction_head/Sigmoid_output_0",  "/ArgMax_output_0"
    int decoderTails(const int& originalImgHeight, const int& originalImgWidth,
        const cv::Mat& Reshape_12_output_0_, const cv::Mat& GreaterOrEqual_output_0_, const cv::Mat& iou_prediction_head_Sigmoid_output_0_, const cv::Mat& ArgMax_output_0_, cv::Mat& mask, std::vector<float>& iou_predictions)
    {
        using namespace dnn::ocvHelper;
        Blob<float> Reshape_12_output_0(Reshape_12_output_0_);
        Blob<float> GreaterOrEqual_output_0(GreaterOrEqual_output_0_);
        Blob<float> iou_prediction_head_Sigmoid_output_0(iou_prediction_head_Sigmoid_output_0_);
        Blob<float> ArgMax_output_0(ArgMax_output_0_);//onnx out int64,but in opencv the data is float still
        auto Slice_8_output_0 = Reshape_12_output_0.sliceSecondDimStep1(0, 1);
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
            Concat_19_output_0.emplace_back(1);
        }
        Concat_19_output_0.emplace_back(Shape_33_output_0[3]);
        Concat_19_output_0.emplace_back(Shape_33_output_0[3]);
        Blob<float>Flatten_output_0 = Slice_6_output_0.flattenAxis2();
        Blob<float>Gather_29_output_0 = Flatten_output_0.gatherDim0(Add_14_output_0);
        Blob<float>Reshape_14_output_0 = Gather_29_output_0.reshape(Concat_19_output_0);
        Blob<float>Unsqueeze_26_output_0 = Reshape_14_output_0.unsqueeze(1);
        auto& Where_8_output_0 = Slice_8_output_0.whereInplaceAndClip(Expand_10_output_0, Unsqueeze_26_output_0, -32, 32);
        mask = Where_8_output_0.convertToMat(256, 256, originalImgHeight, originalImgWidth);

        Blob<float>Gather_31_output_0 = Flatten_1_output_0.gatherDim0(Add_15_output_0);
        Blob<float> Unsqueeze_27_output_0 = Gather_31_output_0.reshape(Shape_39_output_0);
        auto& iou_predictions_blob = Slice_9_output_0.whereInplace(Expand_11_output_0, Unsqueeze_27_output_0);
        iou_predictions = iou_predictions_blob.convertToVec();
        return 0;
    }
    int test()
    {

        cv::Mat img = cv::imread("D:/repo/colmap-third/a.bmp");
        cv::Size originalImgSize = img.size();
        cv::Scalar mean(0.485 * 255, 0.456 * 255, 0.406 * 255);
        cv::Mat imgBlob = cv::dnn::blobFromImage(img, 1. / 57.12, cv::Size(1024, 1024), mean, true);
        std::cout << "load img." << std::endl;
        //return test_dynamic_reshape();
        const int netImgSize = 1024;
        cv::dnn::Net imgEncoderNet = cv::dnn::readNetFromONNX("D:/repo/colmap-third/models/opencv_encoder.onnx");
        std::cout << "load EncoderNet." << std::endl;
        int sz[] = { 1,3,netImgSize,netImgSize };

        imgEncoderNet.setPreferableBackend(cv::dnn::DNN_TARGET_CPU);
        imgEncoderNet.setInput(imgBlob);
        std::vector<cv::Mat> imgEncoderNetOut;
        std::vector<std::string> outLayersNames = { "high_res_feats_0","high_res_feats_1","image_embed" };
        auto start1 = std::chrono::steady_clock::now();
        imgEncoderNet.forward(imgEncoderNetOut, outLayersNames);  // crash here
        auto end1 = std::chrono::steady_clock::now();
        auto elapsed1 = std::chrono::duration_cast<std::chrono::milliseconds>(end1 - start1).count();
        std::cout << "Elapsed time: " << elapsed1 << " ms" << std::endl;
        std::cout << "encode forward ok " << std::endl;


        cv::Mat& high_res_feats_0 = imgEncoderNetOut[0];
        cv::Mat& high_res_feats_1 = imgEncoderNetOut[1];
        cv::Mat& image_embed = imgEncoderNetOut[2];
        //::dnn::ocvHelper::printBlob(high_res_feats_0);
        //::dnn::ocvHelper::printBlob(high_res_feats_1);
        //::dnn::ocvHelper::printBlob(image_embed);

        return 0;

        cv::dnn::Net positionDecoderNet = cv::dnn::readNetFromONNX("D:/repo/colmap-third/models/opencv_decoder.onnx");
        //std::vector<cv::Vec2f>point_coord = { {10., 10.} ,{500., 400.},{200., 600.},{100., 300.},{200., 300.},{1,1} };
        //std::vector<float>point_label = { 1,1,1,1,-1 ,1 };
        std::vector<cv::Vec2f>point_coord = { {983,679} };
        std::vector<float>point_label = { 1 };

        positionDecoderNet.setPreferableBackend(cv::dnn::DNN_TARGET_CPU);
        cv::Mat point_coord_blob;
        cv::Mat point_label_blob;
        cv::Mat inputArrayPlus6;
        dnn::ocvHelper::generPositionBlob(point_coord, point_label, point_coord_blob, point_label_blob, originalImgSize);
        dnn::ocvHelper::generDnnBlob(inputArrayPlus6, { 1,static_cast<int>(point_coord.size()) + 6,1 });
        cv::Mat mask_input;
        dnn::ocvHelper::generDnnBlob(mask_input, { 1, 1, 1024 / 4, 1024 / 4 });
        mask_input.setTo(0);
        cv::Mat has_mask_input;
        dnn::ocvHelper::generDnnBlob(has_mask_input, { 1 });
        has_mask_input.setTo(1);
        cv::Mat orig_im_size;
        dnn::ocvHelper::generDnnBlob(orig_im_size, { 2 }, dnn::ocvHelper::OnnxType::onnx_int32);
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
            auto start2 = std::chrono::steady_clock::now();
            positionDecoderNet.forward(out, outLayersNames);
            auto end2 = std::chrono::steady_clock::now();
            auto elapsed2 = std::chrono::duration_cast<std::chrono::milliseconds>(end2 - start2).count();
            std::cout << "Elapsed time: " << elapsed2 << " ms" << std::endl;
            //printBlob(out[0]);
            //printBlob(out[1]);
            //printBlob(out[2]);
            //printBlob(out[3]);
            std::cout << "forward ok " << std::endl;
            decoderTails(1080, 1920, out[0], out[1], out[2], out[3], mask, iou_predictions);
            std::cout << "done " << std::endl;

            cv::Mat asd2;
            cv::threshold(mask, asd2, 0, 255, cv::THRESH_BINARY);
            asd2.convertTo(asd2, CV_8UC1);
            cv::imwrite("D:/repo/colmap-third/mask.png", asd2);
        }


        return 0;
    }

    const std::vector<int>Sam2::high_res_feats_0_shape = { 1,32,256,256 };
    const std::vector<int>Sam2::high_res_feats_1_shape = { 1, 64, 128, 128 };;
    const std::vector<int>Sam2::image_embed_shape = { 1, 256, 64, 64 };;
	Sam2::Sam2(
		const std::filesystem::path& ncnnEncoderParamPath, const std::filesystem::path& ncnnEncoderBinPath, 
		const std::filesystem::path& onnxDecoderPath)
	{
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
        if (!std::filesystem::exists(onnxDecoderPath))
        {
            LOG_ERR_OUT << "not found : " << onnxDecoderPath;
            return;
        }
		encoderNet.opt.use_vulkan_compute = true;
        encoderNet.opt.num_threads = 8;
		if (encoderNet.load_param(ncnnEncoderParamPath.string().c_str()))
			exit(-1);
		if (encoderNet.load_model(ncnnEncoderBinPath.string().c_str()))
			exit(-1);
        encoderNet.opt.blob_allocator;
        encoderNet.opt.workspace_allocator;
        positionDecoderNet = cv::dnn::readNetFromONNX(onnxDecoderPath.string());
        positionDecoderNet->setPreferableBackend(cv::dnn::DNN_TARGET_CPU);
	}
	bool Sam2::inputImage(const std::filesystem::path& imgPath)
	{
		high_res_feats_0 = cv::Mat();
		high_res_feats_1 = cv::Mat();
		image_embed = cv::Mat();
		if (encoderNet.layers().size()==0)
		{
			LOG_ERR_OUT << "not innitialed!";
			return false;
		}
		cv::Mat img = cv::imread(imgPath.string());
        return inputImage(img);
	}
    bool Sam2::inputImage(const cv::Mat& img)
    {
        high_res_feats_0 = cv::Mat();
        high_res_feats_1 = cv::Mat();
        image_embed = cv::Mat();
        if (img.empty())
        {
            LOG_ERR_OUT << "empty img";
            return false;
        }
        ncnn::Extractor ex_encoder = encoderNet.create_extractor();
        oringalSize = img.size();
        const int netImgSize = 1024;
        ncnn::Mat imgBlob = ncnn::Mat::from_pixels_resize(img.data, ncnn::Mat::PIXEL_BGR2RGB, img.cols, img.rows, netImgSize, netImgSize);
        const float mean_vals[3] = { 0.485 * 256.,0.456 * 256., 0.406 * 256. };
        const float norm_vals[3] = { 0.00390625 / 0.229, 0.00390625 / 0.224, 0.00390625 / 0.225 };
        imgBlob.substract_mean_normalize(mean_vals, norm_vals);
        //ncnnHelper::printBlob(imgBlob);
        ncnn::Mat high_res_feats_0_blob;
        ncnn::Mat high_res_feats_1_blob;
        ncnn::Mat imgEmbedding_blob;
        {
            auto start1 = std::chrono::steady_clock::now();
            ex_encoder.input("image", imgBlob);
            ex_encoder.extract("/Transpose_1_output_0", imgEmbedding_blob);
            ex_encoder.extract("high_res_feats_0", high_res_feats_0_blob);
            ex_encoder.extract("high_res_feats_1", high_res_feats_1_blob);
            auto end1 = std::chrono::steady_clock::now();
            auto elapsed1 = std::chrono::duration_cast<std::chrono::milliseconds>(end1 - start1).count();
            std::cout << "encoder Elapsed time: " << elapsed1 * 0.001 << " s" << std::endl;
            //ncnnHelper::printBlob(high_res_feats_0_blob);
            //ncnnHelper::printBlob(high_res_feats_1_blob);
            //ncnnHelper::printBlob(imgEmbedding_blob);
            dnn::ocvHelper::convertNcnnBlobToOpencv(high_res_feats_0_blob, { 1,32,256,256 }, high_res_feats_0);
            dnn::ocvHelper::convertNcnnBlobToOpencv(high_res_feats_1_blob, { 1, 64, 128, 128 }, high_res_feats_1);
            dnn::ocvHelper::convertNcnnBlobToOpencv(imgEmbedding_blob, { 1, 256, 64, 64, }, image_embed);
            //using namespace sam2::ocvHelper;
            //LOG_OUT << ocvHelper::getBlobShape(high_res_feats_0);
            //LOG_OUT << ocvHelper::getBlobShape(high_res_feats_1);
            //LOG_OUT << ocvHelper::getBlobShape(image_embed);
        }
        //ncnnHelper::writeBlob("../high_res_feats_0_blob.dat",high_res_feats_0_blob);
        //ncnnHelper::writeBlob("../high_res_feats_1_blob.dat",high_res_feats_1_blob);
        //ncnnHelper::writeBlob("../imgEmbedding_blob.dat",imgEmbedding_blob);
        return true;
    }

	bool Sam2::inputHint()
	{
        if (positionDecoderNet == std::nullopt)
        {
            LOG_ERR_OUT << "decoder not innitialed!";
            return false;
        }
        if (high_res_feats_0.empty())
        {
            LOG_ERR_OUT << "high_res_feats_0 empty";
            return false;
        }
        if (high_res_feats_1.empty())
        {
            LOG_ERR_OUT << "high_res_feats_1 empty";
            return false;
        }
        if (image_embed.empty())
        {
            LOG_ERR_OUT << "image_embed empty";
            return false;
        }
		//std::vector<cv::Vec2f>point_coord = { {10., 10.} ,{500., 400.},{200., 600.},{100., 300.},{200., 300.},{1,1} };
		//std::vector<float>point_label = { 1,1,1,1,-1 ,1 };
		std::vector<cv::Vec2f>point_coord = { {1399,586} };
		std::vector<float>point_label = { 1 };
		cv::Mat point_coord_blob;
		cv::Mat point_label_blob;
		cv::Mat inputArrayPlus6;
		dnn::ocvHelper::generPositionBlob(point_coord, point_label, point_coord_blob, point_label_blob, oringalSize);
        dnn::ocvHelper::generDnnBlob(inputArrayPlus6, { 1,static_cast<int>(point_coord.size()) + 6,1 });
		cv::Mat mask_input;
        dnn::ocvHelper::generDnnBlob(mask_input, { 1, 1, 1024 / 4, 1024 / 4 });
		mask_input.setTo(0);
		cv::Mat has_mask_input;
        dnn::ocvHelper::generDnnBlob(has_mask_input, { 1 });
		has_mask_input.setTo(1);
		cv::Mat orig_im_size;
        dnn::ocvHelper::generDnnBlob(orig_im_size, { 2 }, dnn::ocvHelper::OnnxType::onnx_int32);
		//((int*)orig_im_size.data)[0] = oringalSize.width;
		//((int*)orig_im_size.data)[1] = oringalSize.height;
		cv::Mat mask;
		std::vector<float> iou_predictions;
		{
			positionDecoderNet->setInput(high_res_feats_0, "high_res_feats_0");
			positionDecoderNet->setInput(high_res_feats_1, "high_res_feats_1");
			positionDecoderNet->setInput(image_embed, "image_embed");
			positionDecoderNet->setInput(point_coord_blob, "/ScatterND_1_output_0");
			positionDecoderNet->setInput(inputArrayPlus6, "inputArrayPlus6");
			positionDecoderNet->setInput(point_label_blob, "/Unsqueeze_8_output_0");
			positionDecoderNet->setInput(mask_input, "mask_input");
			positionDecoderNet->setInput(has_mask_input, "has_mask_input");
			//positionDecoderNet.setInput(orig_im_size, "orig_im_size");
			std::vector<std::string> layersNames = positionDecoderNet->getLayerNames();
			std::vector<std::string> unconnectedOutLayersNames = positionDecoderNet->getUnconnectedOutLayersNames();
			std::vector<std::string> outLayersNames = {
					"/Reshape_12_output_0","/GreaterOrEqual_output_0","/iou_prediction_head/Sigmoid_output_0","/ArgMax_output_0"
			};
			std::vector<cv::Mat> out;
			auto start2 = std::chrono::steady_clock::now();
			positionDecoderNet->forward(out, outLayersNames);
            //dnn::ocvHelper::printBlob(out[0]);
            //dnn::ocvHelper::printBlob(out[1]);
            //dnn::ocvHelper::printBlob(out[2]);
            //dnn::ocvHelper::printBlob(out[3]);
			auto end2 = std::chrono::steady_clock::now();
			auto elapsed2 = std::chrono::duration_cast<std::chrono::milliseconds>(end2 - start2).count();
            LOG_OUT << "Elapsed time: " << elapsed2*0.001 << " s";
			decoderTails(1080, 1920, out[0], out[1], out[2], out[3], mask, iou_predictions);
            LOG_OUT << "done ";
			cv::Mat asd2;
			cv::threshold(mask, asd2, 0, 255, cv::THRESH_BINARY);
			asd2.convertTo(asd2, CV_8UC1);
			cv::imwrite("../mask.png", asd2);
		}
		return true;
	}
    
    /// input mask not used
    bool Sam2::inputHint(const std::vector<std::pair<int, cv::Point2i>>& hint, cv::Mat& mask)
    {
        if (positionDecoderNet == std::nullopt)
        {
            LOG_ERR_OUT << "decoder not innitialed!";
            return false;
        }
        if (high_res_feats_0.empty())
        {
            LOG_ERR_OUT << "high_res_feats_0 empty";
            return false;
        }
        if (high_res_feats_1.empty())
        {
            LOG_ERR_OUT << "high_res_feats_1 empty";
            return false;
        }
        if (image_embed.empty())
        {
            LOG_ERR_OUT << "image_embed empty";
            return false;
        }
        if (hint.size() == 0)
        {
            LOG_ERR_OUT << "hint empty";
            return false;
        }
        std::vector<cv::Vec2f>point_coord(hint.size());
        std::vector<float>point_label(hint.size());
        for (size_t i = 0; i < hint.size(); i++)
        {
            point_coord[i][0] = hint[i].second.x;
            point_coord[i][1] = hint[i].second.y;
            point_label[i] = hint[i].first;
            LOG_OUT << point_label[i] << " " << point_coord[i];
        }

        cv::Mat point_coord_blob;
        cv::Mat point_label_blob;
        cv::Mat inputArrayPlus6;
        dnn::ocvHelper::generPositionBlob(point_coord, point_label, point_coord_blob, point_label_blob, oringalSize);
        dnn::ocvHelper::generDnnBlob(inputArrayPlus6, { 1,static_cast<int>(point_coord.size()) + 6,1 });
        cv::Mat mask_input;
        dnn::ocvHelper::generDnnBlob(mask_input, { 1, 1, 1024 / 4, 1024 / 4 });
        mask_input.setTo(0);
        cv::Mat has_mask_input;
        dnn::ocvHelper::generDnnBlob(has_mask_input, { 1 });
        has_mask_input.setTo(0);
        cv::Mat orig_im_size;
        dnn::ocvHelper::generDnnBlob(orig_im_size, { 2 }, dnn::ocvHelper::OnnxType::onnx_int32);

        std::vector<float> iou_predictions;
        {
            positionDecoderNet->setInput(high_res_feats_0, "high_res_feats_0");
            positionDecoderNet->setInput(high_res_feats_1, "high_res_feats_1");
            positionDecoderNet->setInput(image_embed, "image_embed");
            positionDecoderNet->setInput(point_coord_blob, "/ScatterND_1_output_0");
            positionDecoderNet->setInput(inputArrayPlus6, "inputArrayPlus6");
            positionDecoderNet->setInput(point_label_blob, "/Unsqueeze_8_output_0");
            positionDecoderNet->setInput(mask_input, "mask_input");
            positionDecoderNet->setInput(has_mask_input, "has_mask_input");
            //positionDecoderNet.setInput(orig_im_size, "orig_im_size");
            std::vector<std::string> layersNames = positionDecoderNet->getLayerNames();
            std::vector<std::string> unconnectedOutLayersNames = positionDecoderNet->getUnconnectedOutLayersNames();
            std::vector<std::string> outLayersNames = {
                    "/Reshape_12_output_0","/GreaterOrEqual_output_0","/iou_prediction_head/Sigmoid_output_0","/ArgMax_output_0"
            };
            std::vector<cv::Mat> out;
            auto start2 = std::chrono::steady_clock::now();
            positionDecoderNet->forward(out, outLayersNames);
            auto end2 = std::chrono::steady_clock::now();
            auto elapsed2 = std::chrono::duration_cast<std::chrono::milliseconds>(end2 - start2).count();
            LOG_OUT << "Elapsed time: " << elapsed2 * 0.001 << " s";
            cv::Mat maskFloat;
            decoderTails(oringalSize.height, oringalSize.width, out[0], out[1], out[2], out[3], maskFloat, iou_predictions);
            LOG_OUT << "done ";
            cv::Mat asd2;
            cv::threshold(maskFloat, asd2, 0, 255, cv::THRESH_BINARY);
            asd2.convertTo(asd2, CV_8UC1);
            asd2.copyTo(mask);
        }
        return true;
    }
    bool Sam2::inputSingleHint(const float& hintx, const float& hinty, const int labelId,const cv::Mat&inPutMask, cv::Mat& mask)
    {
        if (positionDecoderNet == std::nullopt)
        {
            LOG_ERR_OUT << "decoder not innitialed!";
            return false;
        }
        if (high_res_feats_0.empty())
        {
            LOG_ERR_OUT << "high_res_feats_0 empty";
            return false;
        }
        if (high_res_feats_1.empty())
        {
            LOG_ERR_OUT << "high_res_feats_1 empty";
            return false;
        }
        if (image_embed.empty())
        {
            LOG_ERR_OUT << "image_embed empty";
            return false;
        }
        std::vector<cv::Vec2f>point_coord(1);
        std::vector<float>point_label(1);
        point_coord[0][0] = hintx;
        point_coord[0][1] = hinty;
        point_label[0] = labelId;
        LOG_OUT << point_label[0] << " " << point_coord[0];
        cv::Mat point_coord_blob;
        cv::Mat point_label_blob;
        cv::Mat inputArrayPlus6;
        dnn::ocvHelper::generPositionBlob(point_coord, point_label, point_coord_blob, point_label_blob, oringalSize);
        dnn::ocvHelper::generDnnBlob(inputArrayPlus6, { 1,static_cast<int>(point_coord.size()) + 6,1 });
        cv::Mat mask_input;
        dnn::ocvHelper::generDnnBlob(mask_input, { 1, 1, 1024 / 4, 1024 / 4 });
        if (inPutMask.rows == 1024/4 && inPutMask.cols == 1024/4)
        {
            memcpy(mask_input.data, inPutMask.data, sizeof(float) * 65536);
        }
        else
        {
            mask_input.setTo(0);
        }
        
        cv::Mat has_mask_input;
        dnn::ocvHelper::generDnnBlob(has_mask_input, { 1 });
        has_mask_input.setTo(1);

        std::vector<float> iou_predictions;
        {
            positionDecoderNet->setInput(high_res_feats_0, "high_res_feats_0");
            positionDecoderNet->setInput(high_res_feats_1, "high_res_feats_1");
            positionDecoderNet->setInput(image_embed, "image_embed");
            positionDecoderNet->setInput(point_coord_blob, "/ScatterND_1_output_0");
            positionDecoderNet->setInput(inputArrayPlus6, "inputArrayPlus6");
            positionDecoderNet->setInput(point_label_blob, "/Unsqueeze_8_output_0");
            positionDecoderNet->setInput(mask_input, "mask_input");
            positionDecoderNet->setInput(has_mask_input, "has_mask_input");
            std::vector<std::string> layersNames = positionDecoderNet->getLayerNames();
            std::vector<std::string> unconnectedOutLayersNames = positionDecoderNet->getUnconnectedOutLayersNames();
            std::vector<std::string> outLayersNames = {
                    "/Reshape_12_output_0","/GreaterOrEqual_output_0","/iou_prediction_head/Sigmoid_output_0","/ArgMax_output_0"
            };
            std::vector<cv::Mat> out;
            auto start2 = std::chrono::steady_clock::now();
            positionDecoderNet->forward(out, outLayersNames);
            auto end2 = std::chrono::steady_clock::now();
            auto elapsed2 = std::chrono::duration_cast<std::chrono::milliseconds>(end2 - start2).count();
            LOG_OUT << "Elapsed time: " << elapsed2 * 0.001 << " s";
            decoderTails(oringalSize.height, oringalSize.width, out[0], out[1], out[2], out[3], mask, iou_predictions);
            LOG_OUT << "done ";
            cv::Mat asd2;
            cv::threshold(mask, asd2, 0, 255, cv::THRESH_BINARY);
            asd2.convertTo(mask, CV_8UC1);
        }
        return true;
    }
    bool Sam2::serializationFeat(const std::filesystem::path&path)
    {
        cv::dnn::MatShape feat0;
        cv::dnn::MatShape feat1;
        cv::dnn::MatShape imbed;
        cv::dnn::MatShape feat0_shape;
        cv::dnn::MatShape feat1_shape;
        cv::dnn::MatShape imbed_shape;
        std::vector<char> feat0_dat;
        std::vector<char> feat1_dat;
        std::vector<char> imbed_dat;
        dnn::ocvHelper::serializationBlob(high_res_feats_0, feat0_shape, feat0_dat);
        dnn::ocvHelper::serializationBlob(high_res_feats_1, feat1_shape, feat1_dat);
        dnn::ocvHelper::serializationBlob(image_embed, imbed_shape, imbed_dat);
        std::fstream fout(path, std::ios::out | std::ios::binary);
        fout.write((char*)&oringalSize.height, sizeof(int));
        fout.write((char*)&oringalSize.width, sizeof(int));
        fout.write((char*)&feat0_dat[0], sizeof(char) * feat0_dat.size());
        fout.write((char*)&feat1_dat[0], sizeof(char) * feat1_dat.size());
        fout.write((char*)&imbed_dat[0], sizeof(char) * imbed_dat.size());
        fout.close();
        return true;
    }
    bool Sam2::deserializationFeat(const std::filesystem::path& path)
    {
        if (!std::filesystem::exists(path))
        {
            return false;
        }
        try
        {
            if (!high_res_feats_0.empty())
            {
                high_res_feats_0.release();
            }
            if (!high_res_feats_1.empty())
            {
                high_res_feats_1.release();
            }
            if (!image_embed.empty())
            {
                image_embed.release();
            }
            high_res_feats_0.create(high_res_feats_0_shape.size(), &high_res_feats_0_shape[0], CV_32F);
            high_res_feats_1.create(high_res_feats_1_shape.size(), &high_res_feats_1_shape[0], CV_32F);
            image_embed.create(image_embed_shape.size(), &image_embed_shape[0], CV_32F);
            std::fstream fin(path, std::ios::in | std::ios::binary);
            fin.read((char*)&oringalSize.height, sizeof(int));
            fin.read((char*)&oringalSize.width, sizeof(int));
            int dataTotalCnt = high_res_feats_0.dataend- high_res_feats_0.data;
            fin.read((char*)high_res_feats_0.data, sizeof(char) * dataTotalCnt);
            dataTotalCnt = high_res_feats_1.dataend - high_res_feats_1.data;
            fin.read((char*)high_res_feats_1.data, sizeof(char) * dataTotalCnt);
            dataTotalCnt = image_embed.dataend - image_embed.data;
            fin.read((char*)image_embed.data, sizeof(char) * dataTotalCnt);
            fin.close();
            return true;
        }
        catch (const std::exception&)
        {
            return false;
        }
    }


	Sam2::~Sam2()
	{
        positionDecoderNet = std::nullopt;
	}

}

static cv::Mat gui_img;
static cv::Mat gui_mask;
static cv::Mat gui_addWeight;
static std::vector<std::pair<int, cv::Point2i>> gui_hint;
static void sam2_onMouse(int event, int x, int y, int flags, void* sam2Ins)
{
    if (x >= 0 && x < gui_img.cols && y >= 0 && y < gui_img.rows)
    {

    }
    if (event == cv::MouseEventTypes::EVENT_LBUTTONUP )
    {
        LOG_OUT <<"add  " << x << " " << y;
        gui_hint.emplace_back(std::make_pair(1, cv::Point2i(x, y)));
    }
    if (event == cv::MouseEventTypes::EVENT_RBUTTONUP)
    {
        LOG_OUT << "minus  " << x << " " << y;
        gui_hint.emplace_back(std::make_pair(0, cv::Point2i(x, y)));
    }
    if (event == cv::MouseEventTypes::EVENT_MBUTTONUP)
    { 
        LOG_OUT << "clear  ";
        gui_hint.clear();
        gui_img.copyTo(gui_addWeight);
    }
    if (event == cv::MouseEventTypes::EVENT_LBUTTONUP || event == cv::MouseEventTypes::EVENT_RBUTTONUP)
    {
        ((sam2::Sam2*)sam2Ins)->inputHint(gui_hint, gui_mask);
        //cv::imwrite("../mask.png", gui_mask);

        cv::Mat greenMask = cv::Mat::zeros(gui_mask.size(), CV_8UC1);
        cv::Mat blueMask = cv::Mat::zeros(gui_mask.size(), CV_8UC1);
        cv::Mat mask;
        cv::merge(std::vector<cv::Mat>{ blueMask ,greenMask ,gui_mask }, mask);

        float gamma = 0;
        float maskWeight = 0.5;
        cv::addWeighted(gui_img, maskWeight, mask, 1 - maskWeight, gamma, gui_addWeight);
        LOG_OUT << "done.";
    }
    
    

}
int test_sam_gui()
{
    gui_hint.clear();
    std::string imgPath= "../data3/00002.jpg";
    std::filesystem::path featPath = "../data3/00002.samDat";
    gui_img = cv::imread(imgPath);;
    gui_img.copyTo(gui_addWeight);
    sam2::Sam2 sam2Ins("../models/ncnnEncoder.param", "../models/ncnnEncoder.bin", "../models/opencv_decoder.onnx");
    if (std::filesystem::exists(featPath))
    {
        sam2Ins.deserializationFeat(featPath);
    }
    else
    {
        sam2Ins.inputImage(gui_img);
        sam2Ins.serializationFeat(featPath);
    }
    cv::imshow("image", gui_addWeight);
    cv::setMouseCallback("image", sam2_onMouse, &sam2Ins);
    for (;;)
    {
        cv::waitKey(25);
        cv::imshow("image", gui_addWeight);
    }
    return 0;
}
int test_decoder()
{
    std::string imgPath = "../a.bmp";
    cv::Mat img = cv::imread(imgPath);

    cv::Mat  high_res_feats_0, high_res_feats_1, image_embed;
    dnn::ocvHelper::readBlobFile("D:/repo/colmapThird/high_res_feats_0_blob.dat", high_res_feats_0);
    dnn::ocvHelper::readBlobFile("D:/repo/colmapThird/high_res_feats_1_blob.dat", high_res_feats_1);
    dnn::ocvHelper::readBlobFile("D:/repo/colmapThird/imgEmbedding_blob.dat", image_embed, std::vector<int>{1,256,64,64});
    //dnn::ocvHelper::printBlob(high_res_feats_0);
    //dnn::ocvHelper::printBlob(high_res_feats_1);
    //dnn::ocvHelper::printBlob(image_embed);
    cv::dnn::Net  positionDecoderNet = cv::dnn::readNetFromONNX("D:/repo/colmapThird/models/opencv_decoder.onnx");
    positionDecoderNet.setPreferableBackend(cv::dnn::DNN_TARGET_CPU);
    std::vector<cv::Vec2f>point_coord = { {1399,586} };
    std::vector<float>point_label = { 1 };
    cv::Mat point_coord_blob;
    cv::Mat point_label_blob;
    cv::Mat inputArrayPlus6;
    cv::Size oringalSize(1920,1080);
    dnn::ocvHelper::generPositionBlob(point_coord, point_label, point_coord_blob, point_label_blob, oringalSize);
    dnn::ocvHelper::generDnnBlob(inputArrayPlus6, { 1,static_cast<int>(point_coord.size()) + 6,1 });
    cv::Mat mask_input;
    dnn::ocvHelper::generDnnBlob(mask_input, { 1, 1, 1024 / 4, 1024 / 4 });
    mask_input.setTo(0);
    cv::Mat has_mask_input;
    dnn::ocvHelper::generDnnBlob(has_mask_input, { 1 });
    has_mask_input.setTo(1);
    cv::Mat mask;
    std::vector<float> iou_predictions; 
        positionDecoderNet.setInput(high_res_feats_0, "high_res_feats_0");
        positionDecoderNet.setInput(high_res_feats_1, "high_res_feats_1");
        positionDecoderNet.setInput(image_embed, "image_embed");
        positionDecoderNet.setInput(point_coord_blob, "/ScatterND_1_output_0");
        positionDecoderNet.setInput(inputArrayPlus6, "inputArrayPlus6");
        positionDecoderNet.setInput(point_label_blob, "/Unsqueeze_8_output_0");
        positionDecoderNet.setInput(mask_input, "mask_input");
        positionDecoderNet.setInput(has_mask_input, "has_mask_input");
        std::vector<std::string> layersNames = positionDecoderNet.getLayerNames();
        std::vector<std::string> unconnectedOutLayersNames = positionDecoderNet.getUnconnectedOutLayersNames();
        std::vector<std::string> outLayersNames = {
                "/Reshape_12_output_0","/GreaterOrEqual_output_0","/iou_prediction_head/Sigmoid_output_0","/ArgMax_output_0"
        };
        std::vector<cv::Mat> out;
        auto start2 = std::chrono::steady_clock::now();
        positionDecoderNet.forward(out, outLayersNames);

        sam2::decoderTails(1080, 1920, out[0], out[1], out[2], out[3], mask, iou_predictions);
        //dnn::ocvHelper::printBlob(out[0]);
        //dnn::ocvHelper::printBlob(out[1]);
        //dnn::ocvHelper::printBlob(out[2]);
        //dnn::ocvHelper::printBlob(out[3]);
        return 0;
}
int test_multitimes_construction()
{
    sam2::Sam2 sam2Ins("../models/ncnnEncoder.param", "../models/ncnnEncoder.bin", "../models/opencv_decoder.onnx");
    for (size_t i = 0; i < 40; i++)
    {
        sam2Ins.inputImage("../a.bmp");
    }
    return 0;
}
int segmentDirWithLandmarks(const std::filesystem::path& dir)
{
    return 0;
}
int test_mask()
{
    sam2::Sam2 sam2Ins("../models/ncnnEncoder.param", "../models/ncnnEncoder.bin", "../models/opencv_decoder.onnx");
    std::filesystem::path imgPath = "../data3/00002.jpg";
    std::filesystem::path featPath = "../data3/00002.samDat";
    std::filesystem::path outMaskPath = "../data3/00002.bmp";
    cv::Mat img = cv::imread(imgPath.string());
    if (std::filesystem::exists(featPath))
    {
        sam2Ins.deserializationFeat(featPath);
    }
    else
    {
        sam2Ins.inputImage(imgPath);
        sam2Ins.serializationFeat(featPath);
    }
    cv::Mat outMask;
    sam2Ins.inputSingleHint(100, 150, 1,cv::Mat(), outMask);


    cv::Mat mask1024 = cv::Mat::zeros(256, 256, CV_8UC1);
    cv::circle(mask1024,cv::Point(100/4, 150 /4),20,cv::Scalar(1),-1);
    mask1024 = 1 - mask1024;
    mask1024.convertTo(mask1024,CV_32FC1);
    cv::Mat outMask2;
    sam2Ins.inputSingleHint(100, 150, 2,mask1024, outMask2);
    outMask2 = 1 - outMask2;

    cv::imwrite(outMaskPath.string(),outMask*200);
    return 0;
}
int test_sam2()
{
    //return test_multitimes_construction();
    //return test_decoder();
    return test_sam_gui();
    //dnn::test();

    //sam2::ncnnHelper::convertImgToMemFile("../a.bmp");
    //sam2::ncnnHelper::recoverFromMemfile("../a.bmp.dat");
    //return 0;



	sam2::Sam2 sam2Ins("../models/ncnnEncoder.param","../models/ncnnEncoder.bin", "../models/opencv_decoder.onnx");
	sam2Ins.inputImage("../a.bmp");
    sam2Ins.inputHint();
	return 0;
}