#ifndef _OPTIMIZE_MODEL_H_
#define _OPTIMIZE_MODEL_H_
#include <string>
namespace ba
{
    enum OptiModel
    {
        fk1k2 = 1,
        fk1 = 2,
        fcxcyk1 = 3,
        fixcamera1 = 4,//fixed camera = fcxcy
        k1 = 5,//fixed camera = fcxcy
        fcxcy = 6,
    };
    inline int getEachCameraParamCnt(const ba::OptiModel& optiModel)
    {
        int eachCameraParamCnt = 0;
        switch (optiModel)
        {
        case ba::OptiModel::fk1k2:
            eachCameraParamCnt = 3;
            break;
        case ba::OptiModel::fk1:
            eachCameraParamCnt = 2;
            break;
        case ba::OptiModel::fcxcyk1:
            eachCameraParamCnt = 4;
            break;
        case ba::OptiModel::fixcamera1:
            eachCameraParamCnt = 0;    // 
            break;
        case ba::OptiModel::k1:
            eachCameraParamCnt = 1;
            break;
        case ba::OptiModel::fcxcy:
            eachCameraParamCnt = 3;
            break;
        default:
            return -1;
            break;
        }
        return eachCameraParamCnt;
    }
    inline std::string getOptiModelStr(const ba::OptiModel& optiModel)
    {
        int eachCameraParamCnt = 0;
        switch (optiModel)
        {
        case ba::OptiModel::fk1k2:
            return "fk1k2";
        case ba::OptiModel::fk1:
            return "fk1";
        case ba::OptiModel::fcxcyk1:
            return "fcxcyk1";
        case ba::OptiModel::fixcamera1:
            return "fixcamera1";
        case ba::OptiModel::k1:
            return "k1";
        case ba::OptiModel::fcxcy:
            return "fcxcy";
        default:
            return "error  !!!";
        }
        return "error  !!!";
    }
}
#endif // !_OPTIMIZE_MODEL_H_
