# routes/error_handlers.py
from logging import error
from flask import jsonify, render_template, request, current_app
# from app import app
from exceptions.custom_exceptions import *

def register_error_handlers(app):
    """Register global error handlers for the Flask app."""
    
    @app.errorhandler(AppError)
    def handle_app_error(error):
        """Handle custom application errors"""
        current_app.logger.error(f"Application error: {error.message}")
        
        # Return JSON for API requests, HTML for browser requests
        # if request.is_json or request.path.startswith('/api/'):
        return jsonify({
            'success': False,
            'error': error.error_code.lower() if error.error_code else 'application_error',
            'message': error.message
        }), error.status_code
        # else:
        #    return render_template('error.html', 
        #                         error_code=error.status_code,
        #                         error_message=error.message), error.status_code

    @app.errorhandler(ValidationError)
    def handle_validation_error(error):
        current_app.logger.warning(f"Validation error: {error.message} (field={getattr(error, 'field', None)})")
        return jsonify({
            'success': False,
            'msg': error.message,
            'error': 'validation_error',
            'field': getattr(error.field, 'field', None),
        }), error.status_code

    @app.errorhandler(ConfigurationError)
    def handle_configuration_error(error):
        current_app.logger.error(f"Configuration error: {error.message}")
        return jsonify({
            'success': False,
            'msg': error.message,
            'error': 'configuration_error',
        }), error.status_code

    @app.errorhandler(SimulationError)
    def handle_simulation_error(error):
        current_app.logger.error(f"Simulation error: {error.message}")
        return jsonify({
            'success': False,
            'msg': error.message,
            'error': 'simulation_error',
        }), error.status_code

    @app.errorhandler(MetricsError)
    def handle_metrics_error(error):
        current_app.logger.error(f"Metrics error: {error.message}")
        return jsonify({
            'success': False,
            'msg': error.message,
            'error': 'metrics_error',
        }), error.status_code

    @app.errorhandler(GraphNotFoundError)
    def handle_graph_not_found_error(error):
        current_app.logger.error(f"Graph not found error: {error.message}")
        return jsonify({
            'success': False,
            'msg': error.message,
            'error': 'graph_not_found',
        }), error.status_code

    @app.errorhandler(FileUploadError)
    def handle_file_upload_error(error):
        current_app.logger.error(f"File upload error: {error.message}")
        return jsonify({
            'success': False,
            'msg': error.message,
            'error': 'file_upload_error',
        }), error.status_code

    @app.errorhandler(InsufficientResourcesError)
    def handle_insufficient_resources_error(error):
        current_app.logger.error(f"Insufficient resources error: {error.message}")
        return jsonify({
            'success': False,
            'msg': error.message,
            'error': 'insufficient_resources',
        }), error.status_code

    @app.errorhandler(Exception)
    def handle_unexpected_error(error):
        # Catch-all for errors not explicitly handled above
        current_app.logger.exception(f"Unexpected error: {str(error)}")
        return jsonify({
            'success': False,
            'msg': error.message,
            'error': 'internal_server_error',
            "details": str(error)  # you might want to hide details in production
        }), 500

    @app.errorhandler(ValueError)
    def handle_value_error(error):
        current_app.logger.warning(f"Value error: {error.message}")
        return jsonify({
            'success': False,
            'msg': error.message,
            'error': 'value_error',
        }), 400

    @app.errorhandler(TypeError)
    def handle_type_error(error):
        current_app.logger.warning(f"Type error: {error.message}")
        return jsonify({
            'success': False,
            'msg': error.message,
            'error': 'type_error',
        }), 400
    
    @app.errorhandler(OSError)
    def handle_os_error(error):
        current_app.logger.error(f"OS error: {error.message}")
        return jsonify({
            'success': False,
            'msg': error.message,
            'error': 'internal_server_error',
            "details": str(error)  # you might want to hide details in production
        }), 500
    
    @app.errorhandler(RuntimeError)
    def handle_runtime_error(error):
        current_app.logger.error(f"Runtime error: {error.message}")
        return jsonify({
            'success': False,
            'msg': error.message,
            'error': 'runtime_error',
        }), error.status_code
    
    @app.errorhandler(404)
    def not_found(error):
        """Handle 404 errors"""
        current_app.logger.warning(f"404 error: {error}")
        # if request.is_json or request.path.startswith('/api/'):
        return jsonify({
            'success': False,
            'error': 'not_found',
            'message': 'The requested resource was not found.'
        }), 404
        # else:
        #    return render_template('404.html'), 404
    
    @app.errorhandler(500)
    def internal_error(error):
        """Handle 500 errors"""
        current_app.logger.error(f"Internal server error: {error}", exc_info=True)

        # if request.is_json or request.path.startswith('/api/'):
        return jsonify({
            'success': False,
            'error': 'internal_server_error',
            'message': 'An internal server error occurred'
        }), 500
        # else:
        #    return render_template('500.html'), 500

    @app.errorhandler(413)
    def request_entity_too_large(error):
        """Handle file too large errors"""
        current_app.logger.error(f"File too large error: {error}")
        return jsonify({
            'success': False,
            'error': 'file_too_large',
            'message': 'File size exceeds maximum limit'
        }), 413

    @app.errorhandler(429)
    def ratelimit_handler(error):
        """Handle rate limit errors"""
        current_app.logger.warning(f"Rate limit exceeded: {error}")
        return jsonify({
            'success': False,
            'error': 'rate_limit_exceeded',
            'message': 'Too many requests. Please try again later.'
        }), 429