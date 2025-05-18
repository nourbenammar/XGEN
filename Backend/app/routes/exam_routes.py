from flask import Blueprint, request, jsonify
import os
from app.services.exam_analyzer import analyze_exam_file

exam_bp = Blueprint('exam', __name__)
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@exam_bp.route('/analyze-exam', methods=['POST'])
def analyze_exam():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400

    file = request.files['file']
    if not file.filename.lower().endswith(('.pdf', '.docx')):
        return jsonify({'error': 'Invalid file format. Only PDF or DOCX allowed.'}), 400

    path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(path)

    try:
        result = analyze_exam_file(path)
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500


