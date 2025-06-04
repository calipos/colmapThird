#include <algorithm>>
#include <string>
#include <vector>
#include <fstream>
#include "utils.h"
#include "objectPt.h"


std::map<int, std::array<double, 3>> readPoints3DFromTXT(const std::filesystem::path& points3DTXT)
{
	std::map<int, std::array<double, 3>> ret;
	std::map<int, int> voteCnt;
	std::fstream fin(points3DTXT, std::ios::in);
	std::string aline;
	while (std::getline(fin, aline))
	{
		if (aline[0] == '#')
		{
			continue;
		}
		std::vector<std::string>segs = utils::splitString(aline, " ");
		if (segs.size() > 8 && segs.size() % 2 == 0)
		{
			int point3dId = -1;
			if (!utils::stringToNumber<int>(segs[0], point3dId))
			{
				continue;
			}
			std::array<double, 3> xyz;
			if (!utils::stringToNumber<double>(segs[1], xyz[0]))
			{
				continue;
			}
			if (!utils::stringToNumber<double>(segs[2], xyz[1]))
			{
				continue;
			}
			if (!utils::stringToNumber<double>(segs[3], xyz[2]))
			{
				continue;
			}
			int imgTrackCnt = (segs.size() - 8) / 2;
			std::map<int,int>tracksVote;
			for (int i = 0; i < imgTrackCnt; i++)
			{
				int image_id = -1, point2d_idx = -1;
				if (!utils::stringToNumber<int>(segs[i * 2 + 8], image_id))
				{
					break;
				}
				if (!utils::stringToNumber<int>(segs[i * 2 + 9], point2d_idx))
				{
					break;
				}
				if (tracksVote.count(point2d_idx)==0)
				{
					tracksVote[point2d_idx] = 1;
				}
				else
				{
					tracksVote[point2d_idx] += 1;
				}
			}
			std::vector<std::pair<int, int>> name_score_vec(tracksVote.begin(), tracksVote.end());
			sort(name_score_vec.begin(), name_score_vec.end(), [](const auto& a, const auto& b) {return a.second > b.second; });
			if (voteCnt.count(name_score_vec.begin()->first)==0)
			{
				voteCnt[name_score_vec.begin()->first] = name_score_vec.begin()->second;
				ret[name_score_vec.begin()->first] = xyz;
			}
			else if (name_score_vec.begin()->second > voteCnt[name_score_vec.begin()->first])
			{
				voteCnt[name_score_vec.begin()->first] = name_score_vec.begin()->second;
				ret[name_score_vec.begin()->first] = xyz;
			}
		}
	}
	return ret;
}