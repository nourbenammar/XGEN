import os
from flask import Blueprint, jsonify, request
from ..services.syllabusgenerator_services import SyllabusGeneratorService
import fitz

syllabusgenerator_bp = Blueprint('syllabusgen', __name__)

UPLOAD_FOLDER = 'uploaded_pdfs'


@syllabusgenerator_bp.route('', methods=['POST'])  
def generate_syllabus():
    course_title = request.form.get("courseTitle")
    course_code = request.form.get("courseCode")
    he_hours = request.form.get("heHours")
    hne_hours = request.form.get("hneHours")
    ects_credits = request.form.get("ectsCredits")
    module_manager = request.form.get("moduleManager")
    teachers = request.form.get("teachers")
    pedagogical_unit = request.form.get("pedagogicalUnit")
    course_unit = request.form.get("courseUnit")
    prerequisites=request.form.get("prerequisites")
    gradeandoptions=request.form.get("gradeandoptions")
    evaluation_method=request.form.get("evaluationMethod")

    print("Received form data:")
    print("Course Title:", course_title)
    print("Course Code:", course_code)
    print("He Hours:", he_hours)
    print("Hne Hours:", hne_hours)
    print("ECTS Credits:", ects_credits)
    print("Module Manager:", module_manager)
    print("Teachers:", teachers)
    print("Pedagogical Unit:", pedagogical_unit)
    print("Course Unit:", course_unit)
    print("Prerequisites:", prerequisites)
    print("Grade and Options:", gradeandoptions)
    print("Evaluation Method:", evaluation_method)

    uploaded_files = request.files.getlist("pdfFiles")
    saved_file_names = []

    upload_dir = "uploaded_pdfs"
    os.makedirs(upload_dir, exist_ok=True)

    for file in uploaded_files:
        if file and file.filename:
            filepath = os.path.join(upload_dir, file.filename)
            file.save(filepath)
            saved_file_names.append(file.filename)

    print("Saved files:", saved_file_names)

    return jsonify({"message": "Syllabus received", "files": saved_file_names})


@syllabusgenerator_bp.route('/generate', methods=['POST'])
def generate_syllabus_content():
    data = request.get_json()
    label = data.get("label")
    print(label)

    if label == "Course Description":
        description = syllabus_service.generate_manual_description()
        return jsonify({"content": description})
    elif label == "Learning Outcomes":
        ilos = syllabus_service.generate_ilos_from_pdfs(UPLOAD_FOLDER)  # Call function to extract ILOs from PDFs
        return jsonify({"content": "\n".join(ilos)})  
    elif label == "Topics":
        # Call the method to extract topics from PDFs using the service instance
        topics = syllabus_service.generate_topics_from_pdfs(UPLOAD_FOLDER)
        return jsonify({"content": "\n".join(topics)})  # Return the topics as a newline-separated string
    else:
        return jsonify({"error": "Content not found for the given label."}), 404
