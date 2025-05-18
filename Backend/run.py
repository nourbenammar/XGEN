import os
from dotenv import load_dotenv
from flask import Flask
from flask_cors import CORS
from app.routes.syllabus_routes import syllabus_bp
from app.routes.api import api_bp
from app.routes.exam_routes import exam_bp
from app.routes.syllabusgenerator import syllabusgenerator_bp
from app.config import Config
load_dotenv()

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "http://localhost:3000"}})

os.makedirs(Config.UPLOAD_FOLDER, exist_ok=True)

api_key = os.getenv('LLAMA_CLOUD_API_KEY')
print(f"LLAMA_CLOUD_API_KEY: {api_key[:5]}...{api_key[-5:]} if set, else None")

app.register_blueprint(syllabus_bp, url_prefix='/api/syllabus')
app.register_blueprint(api_bp, url_prefix='/api')
app.register_blueprint(exam_bp, url_prefix='/api')
app.register_blueprint(syllabusgenerator_bp, url_prefix='/syllabusgen')
if __name__ == '__main__':
    app.run(debug=True, port=5000)