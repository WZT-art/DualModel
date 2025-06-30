// #include <AsyncLogger.h>

template<typename... Args>
void AsyncLogger::log(LogLevel level, const char* file, int line, Args... args) {
	try {
		std::stringstream ss;
		(ss << ... << args);   // fold expression
		
		// auto& context = getThreadLocalContext();
		logging::attribute_set attrSet;
		attrSet.insert("Severity", logging::attributes::constant<logging::trivial::severity_level>(convertToBoostSeverity(level)));
		logging::record rec = logging::core::get()->open_record(attrSet);
		
		if (rec) {
			rec.attribute_values().insert("File", logging::attributes::make_attribute_value(file));
			rec.attribute_values().insert("Line", logging::attributes::make_attribute_value(line));
			
			// for(const auto& [name, value] : context.temporaryFields) {
			// 	rec.attribute_values().insert(name, value);
			// }
			
			logging::record_ostream strm(rec);
			strm << ss.str();
			strm.flush();
			logging::core::get()->push_record(std::move(rec));
		}
		
		// context.temporaryFields.clear();
	}
	catch (const std::exception& e) {
	std::cerr << "Logging failed: " << e.what() << std::endl;
    }
}

template<typename... Args>
void AsyncLogger::trace(const char* file, int line, Args... args) {
    log(LogLevel::TRACE, file, line, args...);
}

template<typename... Args>
void AsyncLogger::debug(const char* file, int line, Args... args) {
    log(LogLevel::DEBUG, file, line, args...);
}

template<typename... Args>
void AsyncLogger::info(const char* file, int line, Args... args) {
    log(LogLevel::INFO, file, line, args...);
}

template<typename... Args>
void AsyncLogger::warning(const char* file, int line, Args... args) {
    log(LogLevel::WARNING, file, line, args...);
}

template<typename... Args>
void AsyncLogger::error(const char* file, int line, Args... args) {
    log(LogLevel::ERRO, file, line, args...);
}

template<typename... Args>
void AsyncLogger::fatal(const char* file, int line, Args... args) {
    log(LogLevel::FATAL, file, line, args...);
}

// template<typename T>
// void AsyncLogger::addTemporaryField(const std::string& name, const T& value) {
// 	auto& context = getThreadLocalContext();
// 	context.temporaryFields[name] = attrs::make_attribute_value(value);
// }
