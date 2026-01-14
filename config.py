import os
from dotenv import load_dotenv

load_dotenv()
DEBUG = True
class Config:
    """Base configuration."""
    SECRET_KEY = os.getenv('SECRET_KEY', 'your-secret-key-change-this')
    DEBUG = True
    TESTING = False
    
    # Database
    SQLALCHEMY_DATABASE_URI = os.getenv(
        'DATABASE_URL',
        'sqlite:///app.db'
    )
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    
    # Flask Settings
    JSON_SORT_KEYS = False

class DevelopmentConfig(Config):
    """Development configuration."""
    DEBUG = True

class TestingConfig(Config):
    """Testing configuration."""
    TESTING = True
    SQLALCHEMY_DATABASE_URI = 'sqlite:///:memory:'

class ProductionConfig(Config):
    """Production configuration."""
    DEBUG = False
