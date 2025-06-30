#pragma
#ifndef LOGGER_CONFIG_H
#define LOGGER_CONFIG_H

#include <string>
#include <chrono>

enum class LogLevel {
    TRACE,
    DEBUG,
    INFO,
    WARNING,
    ERRO,
    FATAL
};

struct LoggerConfig {
    std::string logPath;                    // 日志文件路径
    std::string fileNamePattern;            // 日志文件名模式
    size_t rotationSize;                    // 日志文件大小限制
    bool dailyRotation;                     // 是否每天轮转
    size_t maxFiles;                        // 最大保留文件数
    std::string formatPattern;              // 日志格式模式
    LogLevel minSeverity;                   // 最小日志级别
    bool consoleOutput;                     // 是否输出到控制台
	std::chrono::seconds flushInterval;		// 刷新间隔
    
    LoggerConfig() : 
	logPath("logs"),
    fileNamePattern("vision_%Y%m%d_%H%M.log"),
    rotationSize(10 * 1024 * 1024),     	// 10MB
    dailyRotation(true),
    maxFiles(10),
    formatPattern(
		"[%TimeStamp(format=\"%Y-%m-%d %H:%M\")%][%ThreadID%][%Severity%][%File%:%Line%] %Message%"
	),
    minSeverity(LogLevel::TRACE),
    consoleOutput(false),
	flushInterval(std::chrono::seconds(3)) {}
};

#endif // LOGGER_CONFIG_H