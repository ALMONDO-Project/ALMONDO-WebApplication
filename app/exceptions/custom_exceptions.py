class AppError(Exception):
    """Base exception class for application errors"""
    def __init__(self, message: str, error_code: str = None, status_code: int = 500): # internal server error
        super().__init__(message)
        self.message = message
        self.error_code = error_code
        self.status_code = status_code

class ValidationError(AppError):
    """Raised when input validation fails"""
    def __init__(self, message: str, field: str = None):
        super().__init__(message, "VALIDATION_ERROR", 400) # ValueError, TypeError --> Bad user input
        self.field = field

class SimulationError(AppError):
    """Raised when simulation fails"""
    def __init__(self, message: str):
        super().__init__(message, "SIMULATION_ERROR", 500) # RunTimeError, OSError --> Internal server error

class ConfigurationError(AppError):
    """Raised when configuration fails"""
    def __init__(self, message: str):
        super().__init__(message, "CONFIGURATION_ERROR", 500) # RunTimeError, OSError --> Internal server error

class MetricsError(AppError):
    """Raised when metrics calculation fails"""
    def __init__(self, message: str):
        super().__init__(message, "METRICS_ERROR", 500) # RunTimeError, OSError --> Internal server error

class GraphNotFoundError(AppError):
    """Raised when graph file is not found"""
    def __init__(self, message: str):
        super().__init__(message, "GRAPH_NOT_FOUND", 404) # KeyError/custom --> Resource not found

class FileUploadError(AppError):
    """Raised when file upload fails"""
    def __init__(self, message: str):
        super().__init__(message, "FILE_UPLOAD_ERROR", 400) # ValueError, TypeError --> Bad user input

class InsufficientResourcesError(AppError):
    """Raised when system resources are insufficient"""
    def __init__(self, message: str):
        super().__init__(message, "INSUFFICIENT_RESOURCES", 507)