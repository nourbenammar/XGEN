from flask import Flask
from flask_cors import CORS
from .config import Config
from flask_sqlalchemy import SQLAlchemy
from flask_migrate import Migrate
from app.config import config
import os
import logging

# Initialize extensions globally (but without app object yet)
db = SQLAlchemy()
migrate = Migrate()

def create_app(config_class=Config):
    app = Flask(__name__)
    app.config.from_object(config_class)
    
    CORS(app)  # Enable CORS for all routes
    
    # Initialize services
    from app.services.llm_service import initialize_llm
    initialize_llm(app)
    
    # Register blueprints
    from app.routes.api import api_bp
    app.register_blueprint(api_bp, url_prefix='/api')
    
    from app.routes.syllabus_routes import syllabus_bp
    app.register_blueprint(syllabus_bp, url_prefix='/api/syllabus')
    
    from app.routes.exam_routes import exam_bp
    app.register_blueprint(exam_bp, url_prefix='/api')

    from app.routes.syllabusgenerator import syllabusgenerator_bp
    app.register_blueprint(syllabusgenerator_bp, url_prefix='/syllabusgen')

    # Configure logging
    log_level = logging.DEBUG if app.config['DEBUG'] else logging.INFO
    logging.basicConfig(level=log_level,
                        format='%(asctime)s %(levelname)s %(name)s %(threadName)s : %(message)s')
    app.logger.setLevel(log_level)
    app.logger.info(f"Flask App '{__name__}' created with '{config_name}' config.")
    app.logger.info(f"Log Level: {logging.getLevelName(app.logger.getEffectiveLevel())}")
    app.logger.info(f"Debug Mode: {app.config['DEBUG']}")

    # Initialize extensions with app
    db.init_app(app)
    migrate.init_app(app, db)
    app.logger.info("Database and Migrate initialized.")

    # --- Import and Instantiate Services INSIDE create_app ---
    # This ensures the 'app' module is loaded before services try to import 'db' from it.
    try:
        app.logger.info("Initializing application services...")
        # Import service classes here
        from .services.embedding_service import EmbeddingService
        from .services.rl_service import RLService
        from .services.retrieval_service import RetrievalService
        from .services.semantic_mapper import SemanticMapper
        from .services.llm_service import LLMService

        # Embedding service
        embedding_service_instance = EmbeddingService(
             model_name=app.config.get('EMBEDDING_MODEL', 'sentence-transformers/all-MiniLM-L6-v2'),
             faiss_index_folder=app.config.get('FAISS_INDEX_FOLDER', 'app/static/faiss_indices')
        )
        app.embedding_service = embedding_service_instance

        # RL Service
        rl_service_instance = RLService()
        app.rl_service = rl_service_instance

        # Retrieval Service
        retrieval_service_instance = RetrievalService(
             embedding_service=embedding_service_instance,
             rl_service=rl_service_instance,
             faiss_index_folder=app.config.get('FAISS_INDEX_FOLDER', 'app/static/faiss_indices')
        )
        app.retrieval_service = retrieval_service_instance

        # Semantic Mapper
        semantic_mapper_instance = SemanticMapper(
             embedding_service=embedding_service_instance,
             threshold=app.config.get('MAPPER_THRESHOLD', 0.5),
             top_n=app.config.get('MAPPER_TOP_N', 2)
        )
        app.semantic_mapper = semantic_mapper_instance

        # LLM Service
        llm_service_instance = LLMService(
             model_name=app.config.get('LLM_MODEL', 'gemini-1.5-flash-latest'),
             rl_service=rl_service_instance,
             max_retries=app.config.get('LLM_MAX_RETRIES', 1)
        )
        app.llm_service = llm_service_instance
        app.logger.info("Application services initialized and attached to app.")

    except Exception as e:
         app.logger.critical(f"CRITICAL ERROR DURING SERVICE INITIALIZATION: {e}", exc_info=True)
         raise # Stop app creation if services fail

    # Register blueprints (AFTER services are attached)
    app.logger.info("Registering blueprints...")
    from .routes.document_routes import document_bp # Relative import
    app.register_blueprint(document_bp)
    from .routes.chat_routes import chat_bp       # Relative import
    app.register_blueprint(chat_bp)
    app.logger.info("Blueprints registered.")

    # Ensure directories exist
    # Use app.config.get for robustness, provide defaults
    upload_folder = app.config.get('UPLOAD_FOLDER', 'app/static/uploads/documents')
    faiss_folder = app.config.get('FAISS_INDEX_FOLDER', 'app/static/faiss_indices')
    model_folder = app.config.get('MODEL_SAVE_DIR', 'app/static/models')
    debug_folder = app.config.get('DEBUG_LOG_DIR', 'app/debug_logs')

    for folder in [upload_folder, faiss_folder, model_folder, debug_folder]:
         try:
              os.makedirs(folder, exist_ok=True)
              app.logger.info(f"Ensured directory exists: {folder}")
         except OSError as error:
              app.logger.error(f"Error creating directory {folder}: {error}")

    # Root redirect
    @app.route('/')
    def index():
        from flask import redirect, url_for
        return redirect(url_for('document.list_documents'))

    app.logger.info("Application setup complete.")

    return app