#ifndef _INTEGRATED_SCAN_LOG_TO_STDOUT_H_
#define _INTEGRATED_SCAN_LOG_TO_STDOUT_H_ 
#include <iostream>
#include <sstream>
#include <memory>
#ifdef _WIN32 
#define _EXPORT_API_ __declspec(dllexport)
#else
#define _EXPORT_API_
#endif 

namespace LOGG
{
    class _EXPORT_API_ console_out
    {
    public:
        console_out(const char* fileName, int lineIdx);
        ~console_out();
        std::stringstream& getStream();
    private:
        std::stringstream logStreamData;
    };
}


class LogMessageVoidify {
public:
    LogMessageVoidify() { }
    // This has to be an operator with a precedence lower than << but
    // higher than ?:
    void operator&(std::ostream&) { }
};


template <typename T, size_t S>
inline constexpr size_t get_file_name_offset(const T(&str)[S], size_t i = S - 1)
{
    return (str[i] == '/' || str[i] == '\\') ? i + 1 : (i > 0 ? get_file_name_offset(str, i - 1) : 0);
}

template <typename T>
inline constexpr size_t get_file_name_offset(T(&str)[1])
{
    return 0;
}

#define LOG_OUT LOGG::console_out(&__FILE__[get_file_name_offset(__FILE__)], __LINE__).getStream() 
#define LOG_WARN_OUT LOGG::console_out(&__FILE__[get_file_name_offset(__FILE__)], __LINE__).getStream() <<"[WARNING]"
#define LOG_ERR_OUT LOGG::console_out(&__FILE__[get_file_name_offset(__FILE__)], __LINE__).getStream() <<"[ERROR]"
#define CHECK(condition) (condition) ? (void)0 : LogMessageVoidify()& LOGG::console_out(&__FILE__[get_file_name_offset(__FILE__)], __LINE__).getStream() <<"[ERROR]"
#define TIME_START(flag)  auto startTime##flag = clock();
#define TIME_END(flag,s) {LOG_OUT  << #s << " cost(s) : " << ((float)clock() - startTime##flag) / CLOCKS_PER_SEC;}
#define API_START(flag)  auto startTime##flag = clock();
#define API_END(flag,s) {LOG_OUT << #s << " cost(s) : " << ((float)clock() - startTime##flag) / CLOCKS_PER_SEC;}
 
#endif // !_INTEGRATED_SCAN_LOG_TO_STDOUT_H_
