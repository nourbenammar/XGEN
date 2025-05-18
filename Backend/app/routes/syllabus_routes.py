from flask import Blueprint, request, jsonify
from werkzeug.utils import secure_filename
import os
import logging
from ..services.syllabus_service import process_syllabus_file, enhance_curriculum
from ..utils.file_handling import allowed_file
from ..config import Config

syllabus_bp = Blueprint('syllabus', __name__)
logger = logging.getLogger(__name__)

@syllabus_bp.route('/upload', methods=['POST'])
def upload_file():
    logger.info("Received file upload request")
    
    try:
        if 'file' not in request.files:
            logger.error("No file part in the request")
            return jsonify({'error': 'No file part in the request'}), 400
        
        file = request.files['file']
        if file.filename == '':
            logger.error("No file selected")
            return jsonify({'error': 'No file selected'}), 400
        
        if not allowed_file(file.filename):
            logger.error(f"Invalid file type: {file.filename}")
            return jsonify({'error': 'File type not allowed. Only PDF, DOC, or DOCX files are accepted.'}), 400

        filename = secure_filename(file.filename)
        filepath = os.path.join(Config.UPLOAD_FOLDER, filename)
        
        logger.info(f"Saving file to {filepath}")
        file.save(filepath)
        
        if not os.path.exists(filepath):
            logger.error(f"Failed to save file: {filepath}")
            return jsonify({'error': 'Failed to save the file'}), 500
        
        logger.info(f"Processing syllabus file: {filepath}")
        syllabus_data = process_syllabus_file(filepath)
        
        logger.info("Enhancing curriculum")
        enhanced_data = enhance_curriculum(syllabus_data)
        
        logger.info("File processed successfully")
        return jsonify({
            'syllabus': syllabus_data,
            'enhancement': enhanced_data
        })
        
    except ValueError as ve:
        error_msg = str(ve)
        if "Invalid authentication token" in error_msg or "LLAMA_CLOUD_API_KEY" in error_msg or "No text" in error_msg:
            error_msg = "Unable to parse the file. Please ensure the file contains text and try again."
        logger.error(f"Client error: {str(ve)}")
        return jsonify({'error': error_msg}), 400
    except Exception as e:
        logger.error(f"Server error processing file: {str(e)}", exc_info=True)
        return jsonify({'error': 'Error processing file. Please try again later.'}), 500
    finally:
        if os.path.exists(filepath):
            try:
                logger.info(f"Removing temporary file: {filepath}")
                os.remove(filepath)
            except Exception as e:
                logger.error(f"Error removing file {filepath}: {str(e)}")