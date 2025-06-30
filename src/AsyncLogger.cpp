#include <boost/log/core.hpp>
#include <boost/log/trivial.hpp>
#include <boost/log/expressions.hpp>
#include <boost/log/sinks/text_file_backend.hpp>
#include <boost/log/sinks/sync_frontend.hpp>
#include <boost/log/sinks/text_ostream_backend.hpp>
#include <boost/log/utility/setup/file.hpp>
#include <boost/log/utility/setup/common_attributes.hpp>
#include <boost/log/utility/setup/formatter_parser.hpp>
#include <boost/log/attributes/current_thread_id.hpp>
#include <boost/log/support/date_time.hpp>
#include <boost/core/null_deleter.hpp>
#include <filesystem>
#include "AsyncLogger.h"

// thread_local AsyncLogger::LogContext AsyncLogger::threadContext_;

BOOST_LOG_ATTRIBUTE_KEYWORD(thread_id, "ThreadID", attrs::current_thread_id::value_type)
BOOST_LOG_ATTRIBUTE_KEYWORD(file_name, "File", std::string)
BOOST_LOG_ATTRIBUTE_KEYWORD(line_num, "Line", int)

AsyncLogger& AsyncLogger::getInstance() {
    static AsyncLogger instance;
    return instance;
}

AsyncLogger::AsyncLogger() {}

void AsyncLogger::init(const LoggerConfig& config) {
    // Creating a log directory
    if (!config.logPath.empty()) {
        fs::path logDir(config.logPath);
        if (!fs::exists(logDir)) {
            fs::create_directories(logDir);
        }
    }

    // Setting the log file name and path
    std::string fullPath = config.logPath + "/" + config.fileNamePattern;

    // Configuration file backend
    auto fileBackend = boost::make_shared<sinks::text_file_backend>(
        keywords::file_name = fullPath,
        keywords::rotation_size = config.rotationSize,
        keywords::time_based_rotation = config.dailyRotation ? 
            sinks::file::rotation_at_time_point(0, 0, 0) : 
            sinks::file::rotation_at_time_point(25, 0, 0),
		keywords::max_files = config.maxFiles
    );

    // Configuring asynchronous sink
    auto sink = boost::make_shared<sink_t>(fileBackend);

    // Setting up the formatter
    // sink->set_formatter(
    //     expr::stream
    //         << expr::format_date_time<boost::posix_time::ptime>("TimeStamp", "%Y-%m-%d %H:%M:%S.%f")
    //         << " [" << expr::attr<attrs::current_thread_id::value_type>("ThreadID") << "]"
    //         << " [" << expr::attr<logging::trivial::severity_level>("Severity") << "]"
    //         << " [" << expr::attr<std::string>("File") << ":" << expr::attr<int>("Line") << "] "
    //         << expr::message
    // );
	
	setupSinkFormatter(sink, config.formatPattern);

    // Setting the filter level
    sink->set_filter(
        logging::trivial::severity >= convertToBoostSeverity(config.minSeverity)
    );

	sink->locked_backend()->auto_flush(true);
	// sink->locked_backend()->set_flush_interval(config.flushInterval);

    // Add sink to core
    logging::core::get()->add_sink(sink);
    
    // The console output
    if (config.consoleOutput) {
        auto consoleSink = initConsoleSink();
        logging::core::get()->add_sink(consoleSink);
    }

    // Adding Common Properties
    logging::core::get()->add_global_attribute("ThreadID", attrs::current_thread_id());
    logging::add_common_attributes();

    sink_ = sink;
}

void AsyncLogger::setupSinkFormatter(boost::shared_ptr<sink_t> sink, const std::string& pattern) {
    // try {
        // sink->set_formatter(logging::parse_formatter(pattern));
    // }
    // catch (const std::exception&) {
        // 如果自定义格式解析失败，使用默认格式
    std::cout<<"custom format failed"<<std::endl;
    sink->set_formatter(
        expr::stream
            << expr::format_date_time<boost::posix_time::ptime>("TimeStamp", "%Y-%m-%d %H:%M:%S")
            // << expr::format_date_time<boost::posix_time::ptime>("TimeStamp", ".%f")
            << " [" << expr::attr<attrs::current_thread_id::value_type>("ThreadID") << "]"
            << " [" << expr::attr<logging::trivial::severity_level>("Severity") << "]"
            << " [" << expr::attr<std::string>("File") << ":" << expr::attr<int>("Line") << "] "
            << expr::message
    );
    // }
}

boost::shared_ptr<sinks::synchronous_sink<sinks::text_ostream_backend>> AsyncLogger::initConsoleSink() {
    auto backend = boost::make_shared<sinks::text_ostream_backend>();
    backend->add_stream(boost::shared_ptr<std::ostream>(&std::cout, boost::null_deleter()));
    
    auto sink = boost::make_shared<sinks::synchronous_sink<sinks::text_ostream_backend>>(backend);
    
    sink->set_formatter(
        expr::stream
            << expr::format_date_time<boost::posix_time::ptime>("TimeStamp", "%Y-%m-%d %H:%M:%S")
            << " [" << expr::attr<attrs::current_thread_id::value_type>("ThreadID") << "]"
            << " [" << expr::attr<logging::trivial::severity_level>("Severity") << "]"
            << " [" << expr::attr<std::string>("File") << ":" << expr::attr<int>("Line") << "] "
            << expr::message
    );
    
    return sink;
}

logging::trivial::severity_level AsyncLogger::convertToBoostSeverity(LogLevel level) {
    switch (level) {
        case LogLevel::TRACE:   return logging::trivial::trace;
        case LogLevel::DEBUG:   return logging::trivial::debug;
        case LogLevel::INFO:    return logging::trivial::info;
        case LogLevel::WARNING: return logging::trivial::warning;
        case LogLevel::ERRO:   return logging::trivial::error;
        case LogLevel::FATAL:   return logging::trivial::fatal;
        default:                return logging::trivial::info;
    }
}

// void AsyncLogger::clearTemporaryFields() {
// 	auto& context = getThreadLocalContext();
// 	context.temporaryFields.clear();
// }

// thread_local AsyncLogger::LogContext& AsyncLogger::getThreadLocalContext() {
// 	return threadContext_;
// }

AsyncLogger::~AsyncLogger() {
    try {
		if (sink_) {
			sink_->flush();
			sink_->stop();
			sink_.reset();
		}
	}
    catch (const std::exception& e) {
        std::cerr << "Error during logger cleanup: " << e.what() << std::endl;
    }	
}