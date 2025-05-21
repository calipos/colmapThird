#ifndef _OPTIMIZE_MODEL_H_
#define _OPTIMIZE_MODEL_H_

namespace ba
{
    enum OptiModel
    {
        fk1k2 = 1,
        fcxcyk1 = 2,
        fk1 = 3,
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
        default:
            return -1;
            break;
        }
        return eachCameraParamCnt;
    }
}
#endif // !_OPTIMIZE_MODEL_H_
