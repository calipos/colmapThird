#include <iomanip>
#include <sstream>
#include <functional>
#include "log.h"
namespace LOGG
{
    console_out::console_out(const char* fileName, int lineIdx)
    {
        logStreamData << "[" << fileName << "][" << lineIdx << "] ";
    }
    console_out::~console_out()
    { 
        const std::string& msg = logStreamData.str();
        std::cout << msg << std::endl;;
    }
    std::stringstream& console_out::getStream()
    {
        return logStreamData;
    }
}