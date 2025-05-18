from openai import OpenAI
import fitz 
from docx import Document
import pandas as pd
import re
from typing import List, Tuple
import os


def ask_mixtral(prompt: str, temperature: float = 0.2) -> str:
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are a helpful AI assistant."},
            {"role": "user", "content": prompt}
        ],
        temperature=temperature,
        max_tokens=2048
    )
    return response.choices[0].message.content.strip()

def load_exam_text(path: str) -> str:
    if path.lower().endswith(".docx"):
        doc = Document(path)
        return "\n".join([p.text for p in doc.paragraphs if p.text.strip()])
    elif path.lower().endswith(".pdf"):
        doc = fitz.open(path)
        return "".join([page.get_text() for page in doc])
    else:
        raise ValueError("Unsupported file format.")

def clean_text(text: str) -> str:
    text = re.sub(r'\n+', '\n', text)
    text = re.sub(r'[ ]+', ' ', text)
    return text.strip()

def extract_questions(text: str) -> str:
    prompt = (
        
    )
    return ask_mixtral(prompt)

def detect_bloom_level(question: str) -> str:
    prompt = (
    )
    return ask_mixtral(prompt).strip()

def match_topic_enhanced(question: str, syllabus_topics: List[str]) -> Tuple[str, float]:
    prompt = (
        
    )
    response = ask_mixtral(prompt).strip()
    match = re.search(r"Topic: (.*?), Confidence: (.*)", response)
    if match:
        topic = match.group(1).strip()
        try:
            confidence = float(match.group(2).strip())
        except ValueError:
            confidence = 0.5
        return topic, confidence
    return "Unknown", 0.5

def smart_weighting(question_text: str, bloom_level: str) -> float:
    if re.match(r'^\d+\.\s+', question_text):
        depth_weight = 1.0
    elif re.match(r'^\([a-z]\)\s+', question_text):
        depth_weight = 0.7
    elif re.match(r'^\([ivxlc]+\)\s+', question_text):
        depth_weight = 0.5
    else:
        depth_weight = 0.5

    bloom_scores = {
        "remember": 0.7, "understand": 0.8, "apply": 1.0,
        "analyze": 1.2, "evaluate": 1.4, "create": 1.5
    }
    bloom_multiplier = bloom_scores.get(bloom_level.lower(), 1.0)
    return depth_weight * bloom_multiplier

def generate_question_feedback(question: str, bloom_level: str, topic: str) -> str:
    prompt = (
       
    )
    return ask_mixtral(prompt).strip()

def generate_overall_feedback(questions: List[str]) -> str:
    sample = questions[:10]
    joined = "\n".join(f"{i+1}. {q}" for i, q in enumerate(sample))
    prompt = (
     
    )
    return ask_mixtral(prompt).strip()

def final_analyzer_agent(exam_path: str) -> dict:
    syllabus = {
        "RL Fundamentals": 0.9,
        "Markov Decision Processes (MDPs)": 1.0,
        "Dynamic Programming (DP)": 1.1,
        "Monte Carlo and TD Learning": 1.1,
        "Model-Free Learning, Q-Learning": 1.2,
        "Deep Reinforcement Learning & DQN": 1.3,
        "RL at Scale & Applications": 1.0,
    }

    text = clean_text(load_exam_text(exam_path))
    extracted_text = extract_questions(text)

    questions_raw = re.findall(r'(?:^\d+\.\s+|^\([a-z]\)\s+|^\([ivxlc]+\)\s+)(.*)', extracted_text, flags=re.MULTILINE)
    questions = [q.strip() for q in questions_raw]

    records, raw_scores = [], []
    for idx, q in enumerate(questions, 1):
        bloom = detect_bloom_level(q)
        topic, confidence = match_topic_enhanced(q, list(syllabus.keys()))
        importance = syllabus.get(topic, 1.0)
        weight = smart_weighting(q, bloom) * importance * confidence
        suggestion = generate_question_feedback(q, bloom, topic)
        raw_scores.append(weight)
        records.append({
            "Question Number": idx,
            "Question Text": q,
            "Bloom Level": bloom,
            "Topic": topic,
            "Confidence": confidence,
            "Weight": weight,
            "Recommendation": suggestion
        })

    total = sum(raw_scores)
    for record in records:
        record["Final Points"] = round((20 * record["Weight"]) / total, 2) if total > 0 else 0.0

    df = pd.DataFrame(records)

    general_info = {
        "total_questions": len(df),
        "average_points": round(df["Final Points"].mean(), 2),
        "top_topic": df["Topic"].mode()[0] if not df.empty else "N/A"
    }

    overall_recommendations = generate_overall_feedback(questions)

    return {
        "general_info": general_info,
        "questions": df.to_dict(orient="records"),
        "recommendations": overall_recommendations
    }

def analyze_exam_file(file_path: str) -> dict:
    return final_analyzer_agent(file_path)

if __name__ == "__main__":

    import sys
    if len(sys.argv) < 2:
        print("Usage: python exam_analyzer.py <exam_file_path>")
    else:
        result = analyze_exam_file(sys.argv[1])
        print("\n--- Summary ---")
        print(result["general_info"])
        print("\n--- Recommendations ---")
        print(result["recommendations"])