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
    spdlog::info("  GET  /              - Frontend (mounted from {})", config.frontend_directory);
    spdlog::info("  GET  /*             - Static files (mounted from {})", config.frontend_directory);
    spdlog::info("  GET  /api/status    - Server status");
    
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
    // Frontend handler is no longer needed with mount point approach
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
}

void ProfileServer::setup_middleware() {
    spdlog::debug("Setting up middleware");
    
    // CORS middleware
    if (config.enable_cors) {
        server->set_pre_routing_handler([](const httplib::Request&, httplib::Response& res) {
            res.set_header("Access-Control-Allow-Origin", "*");
            res.set_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS");
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