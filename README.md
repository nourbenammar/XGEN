# X-GEN: AI-Driven Education Design & Assessment Automation

A structured, dynamic, and personalized education platform developed at **Esprit School of Engineering**.

> This project was created as part of the coursework in **AI for Education** at *Esprit School of Engineering*, to improve the fairness, efficiency, and alignment of academic course design and assessment using cutting-edge AI.

---

## 🌍 Overview

**X-GEN** revolutionizes academic course design by providing educators with intelligent tools to:
- Evaluate and restructure syllabi
- Align assessments with Intended Learning Outcomes (ILOs)
- Automate exam creation and grading criteria
- Promote SMART and Bloom-aligned learning
- Generate structured course content and materials
- Support students via intelligent AI revision tools

---

## 🚀 Features

- 🔍 **Syllabus Evaluation**: Detects inconsistencies and misalignments in learning objectives (ILOs)
- 🧠 **AI-Powered Recommendations**: Improves syllabi and assessments using LLMs (Mistral, LLaMA 3/4, GPT-4, Groq)
- 📝 **Exam Automation**: Generates exams from syllabi, standardizes question complexity and timing
- 📊 **Analytics Dashboard**: Visualizes syllabus structure, Bloom’s taxonomy levels, and ILO coverage
- 🎯 **Learning Outcome Optimization**: Ensures all ILOs are SMART-compliant and Bloom-aligned
- 🏗️ **Syllabus Generator**: Automatically generates structured and pedagogically sound syllabi
- 🧾 **Exam Generation from Syllabus**: Produces question papers aligned with learning outcomes and Bloom levels
- ⚖️ **Question Weighting Engine**: Assigns weights based on complexity and cognitive demand
- 📚 **Course Material Generator**: Creates lecture notes, exercises, and assignments
- 🤖 **Student Revision Chatbot**: Interactive AI assistant to support student review
- 💬 **Conversational AI Support**: Helps educators design and audit courses effectively

---

## 🛠 Tech Stack

### 🎨 Frontend
- [React.js](https://reactjs.org/)
- [TailwindCSS](https://tailwindcss.com/)
- Axios

### 🔧 Backend
- [Flask](https://flask.palletsprojects.com/)
- [FastAPI](https://fastapi.tiangolo.com/) (used for async LLM endpoints)

### 🧠 AI & Orchestration
- [Mistral 7B](https://mistral.ai/)
- [LLaMA 3](https://ai.meta.com/llama/)
- [LLaMA 4](https://llama.meta.com/)
- [Groq API](https://groq.com/)
- [OpenAI GPT-4 / 3.5](https://platform.openai.com/)
- HuggingFace Transformers
- LangChain

### 🧪 DevOps & Tools
- GitHub Education (repo hosting)
- Heroku / DigitalOcean (deployment)
- Postman / Swagger (API testing)
- Git & GitHub for version control


---

## 🚀 Getting Started

### 1. Clone the repository

git clone https://github.com/nourbenammar/XGEN.git

cd XGEN-AI
 
### 2. Frontend Setup (React)

cd frontend

npm install

npm run dev
 
### 3. Backend Setup (Flask)

cd ../backend

python -m venv venv

source venv/bin/activate  # On Windows: venv\Scripts\activate

pip install -r requirements.txt

python app.py


## 🙌 Acknowledgments

# This work was realized by the AImpact Team as part of the AI for Education course at Esprit School of Engineering.

# Team Members:

Malek Gharsallah

Eya Ben Moulehem

Nadia Trabelsi

Fedi Fehmi

Mohamed Iheb Al hadramy

Nour Ben Ammar
