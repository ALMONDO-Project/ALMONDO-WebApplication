import os
"""
# Usage in any route or service file, using current_app.config['name_config_object']
from flask import current_app 
insert into the function: 
upload_folder = current_app.config['UPLOAD_FOLDER']
user_graphs_folder = current_app.config['USER_GRAPHS_FOLDER']
"""

class Config:
    """Base configuration class"""
    
    # Base paths
    DATA_BASE_URL = '../data'
    
    # Flask configurations
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'your-secret-key-here'
    
    # Upload configurations
    UPLOAD_FOLDER = os.path.join(DATA_BASE_URL, 'uploads')
    DOWNLOAD_FOLDER = os.path.join(DATA_BASE_URL, 'downloads')
    RESULTS_FOLDER = os.path.join(DATA_BASE_URL, 'simulation_results')
    
    # Specific graph folders
    GENERATED_GRAPHS_FOLDER = os.path.join(UPLOAD_FOLDER, "graphs", "generated_graphs")
    USER_GRAPHS_FOLDER = os.path.join(UPLOAD_FOLDER, "graphs", "user_graphs")
    DOWNLOAD_GRAPHS_FOLDER = os.path.join(DOWNLOAD_FOLDER, "graphs", "downloaded_graphs")
    
    # File configurations
    ALLOWED_EXTENSIONS = {'csv', 'txt', 'json', 'edgelist', 'png', 'jpg', 'jpeg', 'svg'}
    #MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB max file size
    
    @staticmethod
    def init_app(app):
        """Initialize application with configuration"""
        # Create all necessary directories
        folders_to_create = [
            Config.UPLOAD_FOLDER,
            Config.DOWNLOAD_FOLDER,
            Config.RESULTS_FOLDER,
            Config.GENERATED_GRAPHS_FOLDER,
            Config.USER_GRAPHS_FOLDER,
            Config.DOWNLOAD_GRAPHS_FOLDER
        ]
        
        for folder in folders_to_create:
            os.makedirs(folder, exist_ok=True)
        
        print("All directories created successfully!")

class DevelopmentConfig(Config):
    """Development configuration"""
    DEBUG = True
    TESTING = False

class ProductionConfig(Config):
    """Production configuration"""
    DEBUG = False
    TESTING = False

class TestingConfig(Config):
    """Testing configuration"""
    TESTING = True
    DATA_BASE_URL = './test_data'  # Use different path for testing

# Configuration dictionary
config = {
    'development': DevelopmentConfig,
    'production': ProductionConfig,
    'testing': TestingConfig,
    'default': DevelopmentConfig
}