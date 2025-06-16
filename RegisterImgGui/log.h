#ifndef _INTEGRATED_SCAN_LOG_TO_STDOUT_H_
#define _INTEGRATED_SCAN_LOG_TO_STDOUT_H_ 
#include <iostream>
#include <sstream>
#include <memory>
#include <string>
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
 
#define DEFINE_CHECK_OP_IMPL(name, op)                                       \
  template <typename T1, typename T2>                                        \
  inline const char* name##Impl(const T1& v1, const T2& v2 ) {               \
    if ((v1 op v2)) {                                                        \
      return nullptr;                                                        \
    }                                                                        \
    return "CHECK FAIL";                                                     \
  }                                                                          \
  inline const char* name##Impl(int v1, int v2) {                           \
    return name##Impl<int, int>(v1, v2);                                      \
  }
 
DEFINE_CHECK_OP_IMPL(Check_EQ_, == )
DEFINE_CHECK_OP_IMPL(Check_NE_, != )
DEFINE_CHECK_OP_IMPL(Check_LE_, <= )
DEFINE_CHECK_OP_IMPL(Check_LT_, < )
DEFINE_CHECK_OP_IMPL(Check_GE_, >= )
DEFINE_CHECK_OP_IMPL(Check_GT_, > )

#undef DEFINE_CHECK_OP_IMPL

#  define CHECK_OP_LOG(name, op, val1, val2)                              \
    if (nullptr != Check##name##Impl(val1,   val2,)                            \
        LOG_ERR_OUT;

#define THROW_CHECK_OP__(name, op, val1, val2) \
  CHECK_OP_LOG(name, op, val1, val2)

#define THROW_CHECK_EQ(val1, val2) THROW_CHECK_OP__(_EQ_, ==, val1, val2)
#define THROW_CHECK_NE(val1, val2) THROW_CHECK_OP__(_NE_, !=, val1, val2)
#define THROW_CHECK_LE(val1, val2) THROW_CHECK_OP__(_LE_, <=, val1, val2)
#define THROW_CHECK_LT(val1, val2) THROW_CHECK_OP__(_LT_, <, val1, val2)
#define THROW_CHECK_GE(val1, val2) THROW_CHECK_OP__(_GE_, >=, val1, val2)
#define THROW_CHECK_GT(val1, val2) THROW_CHECK_OP__(_GT_, >, val1, val2)

#define THROW_CHECK_NOTNULL(val) \
  ThrowCheckNotNull(     \
      __FILE__, __LINE__, "'" #val "' Must be non NULL", (val))

#endif // !_INTEGRATED_SCAN_LOG_TO_STDOUT_H_
