import os
from dotenv import load_dotenv


load_dotenv()

class Config:
    LLAMA_CLOUD_API_KEY = os.getenv('LLAMA_CLOUD_API_KEY')
    HUGGINGFACE_TOKEN = os.getenv('HUGGINGFACE_TOKEN')
    MODEL_NAME = os.getenv('MODEL_NAME', 'meta-llama/Llama-3.1-8B-Instruct')
    
    # File upload settings
    BASE_DIR = os.path.abspath(os.path.dirname(__file__))
    UPLOAD_FOLDER = os.path.join(BASE_DIR, 'uploads')
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB file size limit
    ALLOWED_EXTENSIONS = {'pdf', 'doc', 'docx'}  # Allowed file extensions
    
    @staticmethod
    def init_app(app):
        # Create upload folder if it doesn't exist
        if not os.path.exists(Config.UPLOAD_FOLDER):
            os.makedirs(Config.UPLOAD_FOLDER)
     # Database
    SQLALCHEMY_DATABASE_URI = os.environ.get('DATABASE_URL') or \
        'sqlite:///xgen.db'
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    
    # Upload settings
    UPLOAD_FOLDER = os.environ.get('UPLOAD_FOLDER') or 'app/static/uploads/documents'
    MAX_CONTENT_LENGTH = 50 * 1024 * 1024  # 50MB max upload size
    
    # API Keys
    GOOGLE_API_KEY = os.environ.get('GOOGLE_API_KEY')
    HUGGINGFACE_TOKEN = os.environ.get('HUGGINGFACE_TOKEN')

class DevelopmentConfig(Config):
    """Development configuration."""
    DEBUG = True
    TESTING = False

class TestingConfig(Config):
    """Testing configuration."""
    DEBUG = True
    TESTING = True
    SQLALCHEMY_DATABASE_URI = 'sqlite:///:memory:'

class ProductionConfig(Config):
    """Production configuration."""
    DEBUG = False
    TESTING = False
    
    # Override with environment variables in production

# Configuration dictionary
config = {
    'development': DevelopmentConfig,
    'testing': TestingConfig,
    'production': ProductionConfig,
    'default': DevelopmentConfig
}