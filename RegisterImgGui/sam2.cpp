#include "log.h"
//#include "net.h"
//#include "opencv2/opencv.hpp"
#include <string>
#include <filesystem>

namespace sam2
{
	class Sam2
	{
	public:
		Sam2(const std::filesystem::path& ncnnEncoderBeginningParamPath, const std::filesystem::path& ncnnEncoderBeginningBinPath,
			const std::filesystem::path& ncnnEncoderParamPath, const std::filesystem::path& ncnnEncoderBinPath, 
			const std::filesystem::path& onnxDecoderPath);
		~Sam2();

	private:
		//ncnn::Net encoderBeginningNet;
		//ncnn::Net encoderNet;
		//ncnn::Extractor* ex_encoderBeginning;
		//ncnn::Extractor* ex_encoder;
	};

	Sam2::Sam2(
		const std::filesystem::path& ncnnEncoderBeginningParamPath, const std::filesystem::path& ncnnEncoderBeginningBinPath, 
		const std::filesystem::path& ncnnEncoderParamPath, const std::filesystem::path& ncnnEncoderBinPath, 
		const std::filesystem::path& onnxDecoderPath)
	{
		if (!std::filesystem::exists(ncnnEncoderBeginningParamPath))
		{
			LOG_ERR_OUT << "not found : " << ncnnEncoderBeginningParamPath;
			return;
		}
		if (!std::filesystem::exists(ncnnEncoderBeginningBinPath))
		{
			LOG_ERR_OUT << "not found : " << ncnnEncoderBeginningBinPath;
			return;
		}
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



		//encoderBeginningNet.opt.use_vulkan_compute = true;
		//if (encoderBeginningNet.load_param(ncnnEncoderBeginningParamPath.string().c_str()))
		//	exit(-1);
		//if (encoderBeginningNet.load_model(ncnnEncoderBeginningBinPath.string().c_str()))
		//	exit(-1);
		//*ex_encoderBeginning = encoderBeginningNet.create_extractor();
		//encoderNet.opt.use_vulkan_compute = true;
		//if (encoderNet.load_param(ncnnEncoderParamPath.string().c_str()))
		//	exit(-1);
		//if (encoderNet.load_model(ncnnEncoderBinPath.string().c_str()))
		//	exit(-1);
		//*ex_encoder = encoderNet.create_extractor();
	}

	Sam2::~Sam2()
	{
	}
}
int test_sam2()
{
	sam2::Sam2("../models/ncnnEncoderBeginning.param","../models/ncnnEncoderBeginning.bin","../models/ncnnEncoder.param","../models/ncnnEncoder.bin","");
	return 0;
}