#include "profile_server.hpp"
#include "../third_party/spdlog/include/spdlog/spdlog.h"
#include "../third_party/spdlog/include/spdlog/sinks/stdout_color_sinks.h"

namespace server {

ProfileServer::ProfileServer(const ServerConfig& config) 
    : config(config), running(false) {
    
    // Initialize spdlog
    auto console_sink = std::make_shared<spdlog::sinks::stdout_color_sink_mt>();
    auto logger = std::make_shared<spdlog::logger>("server", console_sink);
    logger->set_level(spdlog::level::from_str(config.log_level));
    spdlog::set_default_logger(logger);
    spdlog::set_pattern("[%Y-%m-%d %H:%M:%S.%e] [%n] [%^%l%$] %v");
    
    server = std::make_unique<httplib::Server>();
    setup_handlers();
    setup_middleware();
    setup_routes();
}

ProfileServer::~ProfileServer() {
    stop();
}

bool ProfileServer::start() {
    if (running) {
        spdlog::warn("Server is already running");
        return false;
    }
    
    spdlog::info("Starting BPF Profiler Server on {}:{}", config.host, config.port);
    spdlog::info("Log level: {}", config.log_level);
    spdlog::info("CORS enabled: {}", config.enable_cors);
    spdlog::info("Frontend directory: {}", config.frontend_directory);
    
    // Log available endpoints
    spdlog::info("Available endpoints:");
    spdlog::info("  GET  /*             - Static files (mounted from {})", config.frontend_directory);
    
    running = true;
    
    // server.listen() is blocking, so this will block until server stops
    bool result = server->listen(config.host.c_str(), config.port);
    
    // When we get here, server has stopped
    running = false;
    spdlog::info("Server stopped");
    
    return result;
}

void ProfileServer::stop() {
    if (running && server) {
        spdlog::info("Stopping server...");
        running = false;
        server->stop();
    }
}

void ProfileServer::setup_handlers() {
    spdlog::debug("Setting up handlers");
    
    status_handler = std::make_unique<StatusHandler>();
    analyzer_registry = std::make_unique<AnalyzerHandlerRegistry>();
    register_default_analyzers();
}

void ProfileServer::register_default_analyzers() {
    spdlog::debug("Registering default analyzers");
    analyzer_registry->register_handler(std::make_unique<ProfileAnalyzerHandler>());
    // Add more analyzers here as needed:
    // analyzer_registry->register_handler(std::make_unique<OffCPUAnalyzerHandler>());
    // analyzer_registry->register_handler(std::make_unique<WallClockAnalyzerHandler>());
}

void ProfileServer::setup_routes() {
    spdlog::debug("Setting up routes");
    
    // Mount the frontend directory for static file serving
    // This automatically handles all static files including assets
    if (!server->set_mount_point("/", config.frontend_directory)) {
        spdlog::error("Failed to mount frontend directory: {}", config.frontend_directory);
    } else {
        spdlog::info("Mounted frontend directory: {} at /", config.frontend_directory);
    }
    
    // Set up additional MIME type mappings for modern web assets
    server->set_file_extension_and_mimetype_mapping(".js", "application/javascript");
    server->set_file_extension_and_mimetype_mapping(".mjs", "application/javascript");
    server->set_file_extension_and_mimetype_mapping(".jsx", "application/javascript");
    server->set_file_extension_and_mimetype_mapping(".ts", "application/javascript");
    server->set_file_extension_and_mimetype_mapping(".tsx", "application/javascript");
    server->set_file_extension_and_mimetype_mapping(".css", "text/css");
    server->set_file_extension_and_mimetype_mapping(".woff", "font/woff");
    server->set_file_extension_and_mimetype_mapping(".woff2", "font/woff2");
    server->set_file_extension_and_mimetype_mapping(".svg", "image/svg+xml");
    
    // API routes - these take precedence over static files
    server->Get("/api/status", [this](const httplib::Request& req, httplib::Response& res) {
        status_handler->handle(req, res);
    });
    
    // Analyzer API routes with extensible design
    // POST /api/v1/analyzers/{analyzer_type}/start
    server->Post(R"(/api/v1/analyzers/([^/]+)/start)", [this](const httplib::Request& req, httplib::Response& res) {
        std::string analyzer_type = req.matches[1];
        auto handler = analyzer_registry->get_handler(analyzer_type);
        if (handler) {
            handler->handle_start(req, res);
        } else {
            res.status = 404;
            res.set_content("{\"error\": {\"code\": \"ANALYZER_NOT_SUPPORTED\", \"message\": \"Analyzer type not supported\"}}", "application/json");
        }
    });
    
    // GET /api/v1/analyzers/{analyzer_type}/{session_id}/status
    server->Get(R"(/api/v1/analyzers/([^/]+)/([^/]+)/status)", [this](const httplib::Request& req, httplib::Response& res) {
        std::string analyzer_type = req.matches[1];
        auto handler = analyzer_registry->get_handler(analyzer_type);
        if (handler) {
            handler->handle_status(req, res);
        } else {
            res.status = 404;
            res.set_content("{\"error\": {\"code\": \"ANALYZER_NOT_SUPPORTED\", \"message\": \"Analyzer type not supported\"}}", "application/json");
        }
    });
    
    // GET /api/v1/analyzers/{analyzer_type}/{session_id}/views
    server->Get(R"(/api/v1/analyzers/([^/]+)/([^/]+)/views)", [this](const httplib::Request& req, httplib::Response& res) {
        std::string analyzer_type = req.matches[1];
        auto handler = analyzer_registry->get_handler(analyzer_type);
        if (handler) {
            handler->handle_views(req, res);
        } else {
            res.status = 404;
            res.set_content("{\"error\": {\"code\": \"ANALYZER_NOT_SUPPORTED\", \"message\": \"Analyzer type not supported\"}}", "application/json");
        }
    });
    
    // POST /api/v1/analyzers/{analyzer_type}/{session_id}/stop
    server->Post(R"(/api/v1/analyzers/([^/]+)/([^/]+)/stop)", [this](const httplib::Request& req, httplib::Response& res) {
        std::string analyzer_type = req.matches[1];
        auto handler = analyzer_registry->get_handler(analyzer_type);
        if (handler) {
            handler->handle_stop(req, res);
        } else {
            res.status = 404;
            res.set_content("{\"error\": {\"code\": \"ANALYZER_NOT_SUPPORTED\", \"message\": \"Analyzer type not supported\"}}", "application/json");
        }
    });
    
    // DELETE /api/v1/analyzers/{analyzer_type}/{session_id}
    server->Delete(R"(/api/v1/analyzers/([^/]+)/([^/]+))", [this](const httplib::Request& req, httplib::Response& res) {
        std::string analyzer_type = req.matches[1];
        auto handler = analyzer_registry->get_handler(analyzer_type);
        if (handler) {
            handler->handle_delete(req, res);
        } else {
            res.status = 404;
            res.set_content("{\"error\": {\"code\": \"ANALYZER_NOT_SUPPORTED\", \"message\": \"Analyzer type not supported\"}}", "application/json");
        }
    });
}

void ProfileServer::setup_middleware() {
    spdlog::debug("Setting up middleware");
    
    // CORS middleware
    if (config.enable_cors) {
        server->set_pre_routing_handler([](const httplib::Request&, httplib::Response& res) {
            res.set_header("Access-Control-Allow-Origin", "*");
            res.set_header("Access-Control-Allow-Methods", "GET, POST, DELETE, OPTIONS");
            res.set_header("Access-Control-Allow-Headers", "Content-Type");
            return httplib::Server::HandlerResponse::Unhandled;
        });
        
        // Handle OPTIONS for CORS
        server->Options(".*", [](const httplib::Request&, httplib::Response&) {
            return;
        });
    }
    
    // Logging middleware
    server->set_logger([](const httplib::Request& req, const httplib::Response& res) {
        spdlog::info("{} {} - {}", req.method, req.path, res.status);
    });
    
    // Error handler
    server->set_error_handler([](const httplib::Request&, httplib::Response& res) {
        spdlog::error("Server error: {}", res.status);
        res.set_content("Internal Server Error", "text/plain");
    });
}

} // namespace server 