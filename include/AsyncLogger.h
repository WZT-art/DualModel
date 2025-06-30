#ifndef ASYNC_LOGGER_H
#define ASYNC_LOGGER_H

#include <boost/log/core.hpp>
#include <boost/log/attributes.hpp>
#include <boost/log/attributes/attribute_value.hpp>
#include <boost/log/trivial.hpp>
#include <boost/log/expressions.hpp>
#include <boost/log/sinks/text_file_backend.hpp>
#include <boost/log/sinks/async_frontend.hpp>
#include <boost/log/sinks/sync_frontend.hpp>
#include <boost/log/utility/setup/file.hpp>
#include <boost/log/utility/setup/common_attributes.hpp>
#include <boost/log/sources/severity_logger.hpp>
#include <boost/log/sources/record_ostream.hpp>
#include <boost/core/null_deleter.hpp>
#include <boost/smart_ptr/shared_ptr.hpp>
#include <boost/smart_ptr/make_shared_object.hpp>
#include <boost/date_time/posix_time/posix_time.hpp>
#include <boost/log/support/date_time.hpp>
#include <boost/log/sinks/text_ostream_backend.hpp>
#include <memory>
#include "LoggerConfig.h"

namespace logging = boost::log;
namespace sinks = boost::log::sinks;
namespace expr = boost::log::expressions;
namespace attrs = boost::log::attributes;
namespace keywords = boost::log::keywords;
namespace fs = std::filesystem;

class AsyncLogger {
public:
    static AsyncLogger& getInstance();
    void init(const LoggerConfig& config = LoggerConfig());
    
    template<typename... Args>
    void log(LogLevel level, const char* file, int line, Args... args);
    
    template<typename... Args>
    void trace(const char* file, int line, Args... args);
    
    template<typename... Args>
    void debug(const char* file, int line, Args... args);
    
    template<typename... Args>
    void info(const char* file, int line, Args... args);
    
    template<typename... Args>
    void warning(const char* file, int line, Args... args);
    
    template<typename... Args>
    void error(const char* file, int line, Args... args);
    
    template<typename... Args>
    void fatal(const char* file, int line, Args... args);
	
	/* 临时字段 */
	struct LogContext {
		std::map<std::string, logging::attribute_value> temporaryFields;
	};
	
	// template<typename T>
	// void addTemporaryField(const std::string& name, const T& value);
	
	void clearTemporaryFields();
    
    ~AsyncLogger();

private:
    AsyncLogger();
    AsyncLogger(const AsyncLogger&) = delete;
    AsyncLogger& operator=(const AsyncLogger&) = delete;

    typedef sinks::asynchronous_sink<sinks::text_file_backend> sink_t;
    logging::trivial::severity_level convertToBoostSeverity(LogLevel level);
    boost::shared_ptr<sinks::synchronous_sink<sinks::text_ostream_backend>> initConsoleSink();

    boost::shared_ptr<sink_t> sink_;
	void setupSinkFormatter(boost::shared_ptr<sink_t> sink, const std::string& pattern);
	
	// static thread_local LogContext threadContext_;
    
    // LogContext& getThreadLocalContext();
};

#include "AsyncLogger.inl"

#define LOG_TRACE(...) AsyncLogger::getInstance().trace(__FILE__, __LINE__, __VA_ARGS__)
#define LOG_DEBUG(...) AsyncLogger::getInstance().debug(__FILE__, __LINE__, __VA_ARGS__)
#define LOG_INFO(...) AsyncLogger::getInstance().info(__FILE__, __LINE__, __VA_ARGS__)
#define LOG_WARNING(...) AsyncLogger::getInstance().warning(__FILE__, __LINE__, __VA_ARGS__)
#define LOG_ERROR(...) AsyncLogger::getInstance().error(__FILE__, __LINE__, __VA_ARGS__)
#define LOG_FATAL(...) AsyncLogger::getInstance().fatal(__FILE__, __LINE__, __VA_ARGS__)
// #define LOG_ADD_TEMP_FIELD(name, value) AsyncLogger::getInstance().addTemporaryField(name, value)
// #define LOG_CLEAR_TEMP_FIELDS() AsyncLogger::getInstance().clearTemporaryFields()

#endif // ASYNC_LOGGER_H