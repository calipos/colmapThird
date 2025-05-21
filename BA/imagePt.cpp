#include <fstream>
#include "imagePt.h"
#include "utils.h"
#include "ceres/rotation.h"

namespace col
{
	ImgPt::ImgPt() {}
	ImgPt::ImgPt(const int& imgId_,
		const std::array<double, 4>& wxyz_,
		const std::array<double, 3>& t_,
		const int& cameraId_,
		const std::filesystem::path& imgPath_,
		std::vector<std::array<double, 2>>&& imgPts_)
	{
		imgId = imgId_;
		wxyz = wxyz_;
		t = t_;
		cameraId = cameraId_;
		imgPath = imgPath_;
		imgPts = imgPts_;

		ceres::QuaternionToAngleAxis<double>(&wxyz[0], &r[0]);
	}
}
std::vector<col::ImgPt> readImgPtsFromTXT(const std::filesystem::path& cameraTXT)
{
	std::vector<col::ImgPt> ret;
	std::fstream fin(cameraTXT, std::ios::in);
	std::string aline;
	std::string aline2;
	while (std::getline(fin, aline))
	{
		if (aline[0] == '#')
		{
			continue;
		}
		col::ImgPt thisLine;
		std::vector<std::string>segs = utils::splitString(aline, " ");
		if (segs.size() >= 10)
		{
			std::getline(fin, aline2);
			if (!utils::stringToNumber<int>(segs[0], thisLine.imgId))
			{
				continue;
			}
			if (!utils::stringToNumber<double>(segs[1], thisLine.wxyz[0]))
			{
				continue;
			}
			if (!utils::stringToNumber<double>(segs[2], thisLine.wxyz[1]))
			{
				continue;
			}
			if (!utils::stringToNumber<double>(segs[3], thisLine.wxyz[2]))
			{
				continue;
			}
			if (!utils::stringToNumber<double>(segs[4], thisLine.wxyz[3]))
			{
				continue;
			}
			if (!utils::stringToNumber<double>(segs[5], thisLine.t[0]))
			{
				continue;
			}
			if (!utils::stringToNumber<double>(segs[6], thisLine.t[1]))
			{
				continue;
			}
			if (!utils::stringToNumber<double>(segs[7], thisLine.t[2]))
			{
				continue;
			}
			if (!utils::stringToNumber<int>(segs[8], thisLine.cameraId))
			{
				continue;
			}
			ceres::QuaternionToAngleAxis<double>(&thisLine.wxyz[0], &thisLine.r[0]);
			std::string name = segs[9];
			for (int i = 10; i < segs.size(); i++)
			{
				name += " ";
				name += segs[i];
			}
			thisLine.imgPath = name;
		}
		segs = utils::splitString(aline2, " ");
		thisLine.imgPts.resize(segs.size() / 3);
		if (segs.size() > 0 && segs.size()%3==0)
		{
			for (int i = 0; i < segs.size()/3; i++)
			{
				utils::stringToNumber<double>(segs[3*i+0], thisLine.imgPts[i][0]);
				utils::stringToNumber<double>(segs[3*i+1], thisLine.imgPts[i][1]);
			}
		}
		ret.emplace_back(thisLine);
	}
	return ret;
}