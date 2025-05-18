import json
import re
import time
import ftfy
import os
import logging
from langdetect import detect
from llama_parse import LlamaParse
from docx import Document
import PyPDF2
from ..utils.file_handling import allowed_file
from .llm_service import generate_response

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

def clean_llm_output(raw_text):
    """Clean LLM output and extract JSON if present."""
    # Remove instruction tags and control characters
    cleaned = re.sub(r'(\[INST\]|\[/INST\]|\s*<\\/s>\s*)', '', raw_text)
    cleaned = re.sub(r'[\\x00-\\x1F\\x7F]', '', cleaned).strip()
    
    # Find JSON content within text
    brace_count = 0
    start_idx = None
    json_str = ""
    for i, char in enumerate(cleaned):
        if char == '{':
            if brace_count == 0:
                start_idx = i
            brace_count += 1
        elif char == '}':
            brace_count -= 1
            if brace_count == 0 and start_idx is not None:
                json_str = cleaned[start_idx:i+1]
                break
                
    # Validate JSON
    if json_str:
        try:
            parsed = json.loads(json_str)
            required_keys = {"course_info", "staff", "content", "assessment"}
            if not all(key in parsed for key in required_keys):
                logger.warning("Incomplete JSON structure. Missing keys: %s", 
                              required_keys - set(parsed.keys()))
        except json.JSONDecodeError as e:
            logger.error("JSON decode failed in clean_llm_output: %s", e)
            json_str = ""
            
    return json_str

def build_bloom_taxonomy_prompt(course_title, outcomes, lang="en"):
    """Build a prompt to improve learning outcomes using Bloom's Taxonomy."""
    # Bloom's taxonomy verbs for English
    verbs_en = {
        "remember": "list, define, recall, identify, name, recognize, reproduce, state",
        "understand": "describe, explain, summarize, interpret, classify, compare, discuss, translate",
        "apply": "implement, use, demonstrate, solve, calculate, execute, illustrate, practice",
        "analyze": "differentiate, compare, organize, attribute, deconstruct, examine, test, question",
        "evaluate": "assess, justify, critique, judge, defend, recommend, prioritize, verify",
        "create": "design, formulate, construct, invent, develop, compose, plan, produce"
    }
    
    # Bloom's taxonomy verbs for French
    verbs_fr = {
        "se_souvenir": "énumérer, définir, reconnaître, identifier, nommer, mémoriser, reproduire, citer",
        "comprendre": "décrire, expliquer, résumer, interpréter, classifier, comparer, discuter, traduire",
        "appliquer": "utiliser, démontrer, résoudre, calculer, exécuter, illustrer, pratiquer, mettre en œuvre",
        "analyser": "différencier, comparer, organiser, attribuer, déconstruire, examiner, tester, questionner",
        "évaluer": "évaluer, justifier, critiquer, juger, défendre, recommander, hiérarchiser, vérifier",
        "créer": "concevoir, formuler, construire, inventer, développer, composer, planifier, produire"
    }

    if lang == "en":
        # English prompt
        prompt = (
           
        )
        
        for level, verbs in verbs_en.items():
            prompt += f"- {level.capitalize()}: {verbs}\n"
            
        prompt += f"\nOriginal Learning Outcomes:\n"
        for i, outcome in enumerate(outcomes, 1):
            prompt += f"{i}. {outcome}\n"
            
        prompt += (
        )
    else:
        # French prompt
        prompt = (
            
        )
        
        for level, verbs in verbs_fr.items():
            prompt += f"- {level.capitalize().replace('_', ' ')}: {verbs}\n"
            
        prompt += f"\nRésultats d'apprentissage originaux:\n"
        for i, outcome in enumerate(outcomes, 1):
            prompt += f"{i}. {outcome}\n"
            
        prompt += (
        )
    
    return prompt

def text_to_syllabus(text, language="english", verbose=True, max_retries=2):
    """Parse syllabus text into structured data."""
    try:
        # Input validation
        if not text.strip():
            raise ValueError("Input text cannot be empty.")
        if len(text) > 100000:
            raise ValueError("Input text is too long. Please provide a shorter syllabus.")
        if len(text.split()) < 50:
            raise ValueError("Input syllabus is too short or lacks sufficient content.")

        logger.debug(f"Text length: {len(text)} chars, {len(text.split())} words")

        start_time = time.time()
        base_structure = {
            "course_info": {
                "code": None,
                "title": None,
                "credits": {
                    "taught_hours": None,
                    "independent_hours": None,
                    "ects": None
                },
                "department": None
            },
            "staff": {
                "instructors": [],
                "assistants": []
            },
            "content": {
                "objectives": [],
                "prerequisites": [],
                "schedule": []
            },
            "assessment": {
                "methods": None,
                "breakdown": []
            }
        }

        def clean_text(text):
            """Clean and normalize syllabus text."""
            cleaned = ftfy.fix_text(text)
            cleaned = re.sub(r'\s+', ' ', cleaned).strip()
            cleaned = re.sub(r'espric', 'esprit', cleaned, flags=re.IGNORECASE)
            cleaned = re.sub(r'\bMS-0821h30h1\b', 'MS-08', cleaned, flags=re.IGNORECASE)
            cleaned = re.sub(r'(\d+)[hH](?!\w)', r'\1h', cleaned)
            cleaned = re.sub(r'(\d+),(\d+)h', r'\1.\2h', cleaned)
            cleaned = re.sub(r'(\d+\.\d+)h', lambda m: f"{float(m.group(1))}h", cleaned)
            cleaned = re.sub(r'(\d+)-(\d+)h', lambda m: f"{m.group(1)}h-{m.group(2)}h", cleaned)
            cleaned = re.sub(r'\b(\d+)\s*(hours|heures|heures\s*de\s*cours|heures\s*TD|heures\s*TP|heures\s*CM)\b', r'\1h', cleaned, flags=re.IGNORECASE)
            cleaned = re.sub(r'\b(TD|TP|CM)\b', lambda m: f"{m.group(1).upper()}: ", cleaned, flags=re.IGNORECASE)
            cleaned = re.sub(r'^\s*[-*•]\s*|\d+\.\s*', ' ', cleaned, flags=re.MULTILINE)
            return cleaned

        cleaned_text = clean_text(text)
        if verbose:
            logger.debug(f"Raw Input Text (length: {len(text)}): {text[:500]}...")
            logger.debug(f"Cleaned Input Text (length: {len(cleaned_text)}): {cleaned_text[:500]}...")

        example_json = json.dumps(base_structure, indent=4, ensure_ascii=False)
        

        instruction_intro = (
        )

        prompt = f"""{instruction_intro}

**Instructions** :
1. {"Output only a valid JSON object, enclosed in {{}} — no extra text or markdown." if language.lower() == "english" else "Retournez uniquement un objet JSON valide, entouré de {{}} — pas de texte supplémentaire ou de markdown."}
2. {"Follow this structure exactly:" if language.lower() == "english" else "Suivez exactement cette structure :"}
{french_example_json if language.lower() == "french" else example_json}
3. {"If information is missing, use null." if language.lower() == "english" else "Si une information est manquante, utilisez null."}
4. {"Convert all time formats and use measurable verbs in objectives." if language.lower() == "english" else "Convertissez tous les formats de temps et utilisez des verbes mesurables dans les objectifs."}
5. {"Return only the JSON object." if language.lower() == "english" else "Retournez uniquement l'objet JSON."}

**{"Syllabus Text" if language.lower() == "english" else "Texte du Syllabus"}** :
{cleaned_text}

**{"Output" if language.lower() == "english" else "Sortie"}** : {"JSON only please." if language.lower() == "english" else "JSON uniquement, s'il vous plaît."}
"""
        if verbose:
            logger.debug("Prompt Sent to LLM: %s", prompt)

        def extract_json_block(response):
            """Extract JSON block from response text."""
            match = re.search(r'({.*})', response, re.DOTALL)
            if match:
                return match.group(1).strip()
            return response.strip()

        parsed = base_structure.copy()
        for attempt in range(max_retries):
            try:
                response = generate_response(
                    prompt,
                    max_new_tokens=4000,
                    temperature=0.2,
                    top_p=0.9
                )
                if verbose:
                    logger.debug(f"Attempt {attempt + 1}/{max_retries} - Raw LLM Response: {response[:500]}...")

                cleaned_response = extract_json_block(response)
                if verbose:
                    logger.debug(f"Cleaned Response: {cleaned_response[:500]}...")

                parsed_json = json.loads(cleaned_response)
                for section in base_structure:
                    if section not in parsed_json:
                        parsed_json[section] = base_structure[section]
                    elif isinstance(base_structure[section], dict) and isinstance(parsed_json[section], dict):
                        for subkey in base_structure[section]:
                            if subkey not in parsed_json[section]:
                                parsed_json[section][subkey] = base_structure[section][subkey]
                parsed = parsed_json
                break
            except json.JSONDecodeError as e:
                logger.error(f"JSON decode failed on attempt {attempt + 1}: {e}")
                logger.error(f"Error at line {e.lineno}, column {e.colno}: {e.msg}")
                logger.error("Partial content: %s", cleaned_response[:300] + "..." if len(cleaned_response) > 300 else cleaned_response)
                if attempt < max_retries - 1:
                    prompt += f"\n\n**{'Note' if language.lower() == 'english' else 'Note'}** : {'Ensure the output is a valid JSON object with proper brackets and syntax. Avoid extra text or incomplete structures.' if language.lower() == 'english' else 'Assurez-vous que la sortie est un objet JSON valide avec des crochets et une syntaxe corrects. Évitez tout texte supplémentaire ou structures incomplètes.'}"
            except ValueError as e:
                logger.error(f"Error in text_to_syllabus attempt {attempt + 1}: {str(e)}")
                if "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" in str(e) or "Groq client not initialized" in str(e):
                    raise ValueError("Groq API key is missing or invalid. Please set a valid GROQ_API_KEY environment variable.")
                if attempt == max_retries - 1:
                    raise ValueError(f"Failed to generate valid JSON after {max_retries} attempts: {str(e)}")
            except Exception as e:
                logger.error(f"Unexpected error in text_to_syllabus attempt {attempt + 1}: {str(e)}")
                if attempt == max_retries - 1:
                    raise ValueError(f"Failed to generate valid JSON after {max_retries} attempts: {str(e)}")

        parsed["processing_metadata"] = {
            "language": language,
            "processing_time": f"{(time.time() - start_time):.2f}s",
            "attempts_used": attempt + 1,
            "source": "llm_extraction_v4",
            "version": "2025.05"
        }

        return json.dumps(parsed, indent=2, ensure_ascii=False)
    except Exception as e:
        logger.error(f"Error in text_to_syllabus: {str(e)}")
        raise

def fallback_parse_file(file_path):
    """Parse file content when LlamaParse is not available."""
    logger.info(f"Attempting fallback parsing for file: {file_path}")
    try:
        if file_path.lower().endswith('.docx'):
            doc = Document(file_path)
            full_text = "\n".join([para.text for para in doc.paragraphs if para.text.strip()])
            if not full_text.strip():
                logger.error("No text found in .docx file")
                raise ValueError("No text found in .docx file")
            logger.info(f"Successfully parsed .docx file: {file_path}")
            return full_text
        elif file_path.lower().endswith('.pdf'):
            with open(file_path, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                if reader.is_encrypted:
                    logger.error("PDF is encrypted")
                    raise ValueError("Cannot parse encrypted PDF")
                full_text = "\n".join([page.extract_text() or "" for page in reader.pages])
                if not full_text.strip():
                    logger.error("No text found in .pdf file")
                    raise ValueError("No text found in .pdf file")
                logger.info(f"Successfully parsed .pdf file: {file_path}")
                return full_text
        else:
            logger.error("Unsupported file type for fallback parsing")
            raise ValueError("Unsupported file type for fallback parsing")
    except Exception as e:
        logger.error(f"Fallback parsing failed: {str(e)}")
        raise ValueError(f"Fallback parsing failed: {str(e)}")

def process_syllabus_file(file_path):
    """Process a syllabus file and extract structured data."""
    logger.info(f"Processing file: {file_path}")
    
    # Get API key for LlamaParse
    api_key = os.getenv("LLAMA_CLOUD_API_KEY")
    logger.info(f"LLAMA_CLOUD_API_KEY: {'set' if api_key else 'not set'}")
    
    full_text = None
    if api_key:
        try:
            logger.info(f"Parsing file with LlamaParse: {file_path}")
            parser = LlamaParse(
                api_key=api_key,
                result_type="markdown"
            )
            document = parser.load_data(file_path)
            full_text = "\n\n".join([doc.text for doc in document])
            logger.info(f"LlamaParse successful for file: {file_path}")
        except Exception as e:
            logger.warning(f"LlamaParse failed: {str(e)}. Attempting fallback parsing.")
    
    # If LlamaParse failed or no API key, try fallback
    if full_text is None:
        try:
            full_text = fallback_parse_file(file_path)
        except Exception as e:
            logger.error(f"Failed to parse file {file_path}: {str(e)}")
            raise ValueError(f"Failed to parse the file: {str(e)}")

    # Check if text is empty
    if not full_text.strip():
        logger.error(f"No text extracted from file: {file_path}")
        raise ValueError("No text could be extracted from the file")

    # Log extracted text for debugging
    logger.debug(f"Extracted text (first 500 chars): {full_text[:500]}")

    # Detect language
    try:
        detected_language = detect(full_text)
        lang_param = "french" if detected_language.startswith("fr") else "english"
        logger.info(f"Detected language: {lang_param}")
    except Exception as e:
        logger.warning(f"Language detection failed: {str(e)}. Defaulting to English.")
        lang_param = "english"
    
    # Clean text
    cleaned_text = re.sub(r'\s+', ' ', full_text).strip()
    
    # Generate structured output
    logger.info("Generating structured output from text")
    try:
        structured_output = text_to_syllabus(
            cleaned_text, 
            lang_param, 
            verbose=True
        )
        return json.loads(structured_output)
    except Exception as e:
        logger.error(f"Failed to generate structured output: {str(e)}")
        raise ValueError(f"Failed to generate structured output: {str(e)}")

def clean_json_response(response):
    """Clean and extract JSON from LLM response."""
    logger.info("Cleaning JSON response")
    try:
        match = re.search(r'({[\s\S]*})', response)
        if match:
            json_str = match.group(1)
            try:
                return json.loads(json_str)
            except json.JSONDecodeError:
                json_str = re.sub(r'[\n\r\t]', ' ', json_str)
                json_str = re.sub(r',\s*}', '}', json_str)
                json_str = re.sub(r',\s*]', ']', json_str)
                try:
                    return json.loads(json_str)
                except json.JSONDecodeError as e:
                    logger.error(f"Failed to parse JSON response: {e}")
                    return None
        logger.warning("No valid JSON found in response")
        return None
    except Exception as e:
        logger.error(f"Error cleaning JSON response: {str(e)}")
        return None

def improve_learning_outcomes(syllabus_data):
    """Extract and improve learning outcomes using Bloom's Taxonomy."""
    logger.info("Improving learning outcomes")
    try:
        # Extract course title and objectives
        course_title = syllabus_data.get("course_info", {}).get("title", "Unknown Course")
        objectives = syllabus_data.get("content", {}).get("objectives", [])
        
        # Determine language for prompt
        lang = "fr" if syllabus_data.get("processing_metadata", {}).get("language", "english") == "french" else "en"
        
        if not objectives:
            logger.warning("No objectives found in syllabus data")
            return {"original_outcomes": [], "improved_outcomes": [], "bloom_analysis": []}
        
        # Generate prompt for improving learning outcomes
        outcomes_prompt = build_bloom_taxonomy_prompt(course_title, objectives, lang=lang)
        logger.debug(f"Outcomes improvement prompt: {outcomes_prompt[:500]}...")
        
        # Call LLM to improve outcomes
        outcomes_response = generate_response(
            outcomes_prompt,
            max_new_tokens=2000,
            temperature=0.2,
            top_p=0.9
        )
        logger.debug(f"Outcomes improvement response: {outcomes_response[:500]}...")
        
        # Extract JSON from response
        json_match = re.search(r'```json\s*({[\s\S]*?})\s*```', outcomes_response)
        if json_match:
            try:
                outcomes_json = json.loads(json_match.group(1))
                return outcomes_json
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse outcomes JSON: {e}")
        
        # If JSON extraction fails, try to parse it manually
        improved_outcomes = []
        bloom_analysis = []
        
        # Look for improved outcomes section
        if lang == "en":
            match = re.search(r'Improved Outcomes:?\n([\s\S]*?)(?:\n\n|$)', outcomes_response)
        else:
            match = re.search(r'Résultats d\'apprentissage améliorés :?\n([\s\S]*?)(?:\n\n|$)', outcomes_response)
            
        if match:
            outcomes_text = match.group(1).strip()
            improved_outcomes = [line.strip('- ').strip() for line in outcomes_text.split('\n') if line.strip().startswith('-')]
        
        return {
            "original_outcomes": objectives,
            "improved_outcomes": improved_outcomes,
            "bloom_analysis": bloom_analysis
        }
        
    except Exception as e:
        logger.error(f"Error improving learning outcomes: {str(e)}")
        return {"original_outcomes": objectives, "improved_outcomes": [], "bloom_analysis": []}

def enhance_curriculum(syllabus_data):
    """Enhance curriculum with detailed Bloom's analysis, misalignments, and suggestions."""
    logger.info("Enhancing curriculum")
    try:
        # Improve learning outcomes
        outcomes_data = improve_learning_outcomes(syllabus_data)
        
        # Prepare syllabus data for analysis
        syllabus_json = json.dumps(syllabus_data, indent=2, ensure_ascii=False)
        
        # Define Bloom's Taxonomy for reference (English names)
        blooms_taxonomy = {
            "Remember": "list, define, recall, identify, name, recognize, reproduce, state",
            "Understand": "describe, explain, summarize, interpret, classify, compare, discuss, translate",
            "Apply": "implement, use, demonstrate, solve, calculate, execute, illustrate, practice",
            "Analyze": "differentiate, compare, organize, attribute, deconstruct, examine, test, question",
            "Evaluate": "assess, justify, critique, judge, defend, recommend, prioritize, verify",
            "Create": "design, formulate, construct, invent, develop, compose, plan, produce"
        }
        blooms_json = json.dumps(blooms_taxonomy, indent=2, ensure_ascii=False)
        
        # Define mapping for French Bloom's levels to English
        french_to_english_bloom = {
            "se_souvenir": "Remember",
            "comprendre": "Understand",
            "appliquer": "Apply",
            "analyser": "Analyze",
            "évaluer": "Evaluate",
            "créer": "Create",
            "Connaissance": "Remember",
            "Compréhension": "Understand",
            "Application": "Apply",
            "Analyse": "Analyze",
            "Évaluation": "Evaluate",
            "Création": "Create"
        }
        
        # Determine language for prompt
        lang = "fr" if syllabus_data.get("processing_metadata", {}).get("language", "english") == "french" else "en"
        
        # Generate enhanced curriculum analysis prompt
        # prompt = 
       
        # Parse and validate the response
        parsed_response = clean_json_response(response)
        if not parsed_response:
            logger.error("Failed to generate valid JSON response from the model")
            raise ValueError("Failed to generate valid JSON response from the model")
        
        # Add improved outcomes data
        parsed_response["improved_outcomes"] = outcomes_data.get("improved_outcomes", [])
        parsed_response["original_outcomes"] = outcomes_data.get("original_outcomes", [])
        parsed_response["bloom_outcome_analysis"] = outcomes_data.get("bloom_analysis", [])
        
        # Validate the response structure
        required_keys = {"bloom_analysis", "misalignments", "suggestions", "scoring"}
        if not all(key in parsed_response for key in required_keys):
            logger.warning("Incomplete JSON structure. Missing keys: %s", 
                          required_keys - set(parsed_response.keys()))
            # Fill missing keys with defaults
            for key in required_keys:
                if key not in parsed_response:
                    parsed_response[key] = {} if key != "suggestions" else []

        # Ensure suggestions meet minimum count
        if len(parsed_response.get("suggestions", [])) < 5:
            logger.warning("Fewer than 5 suggestions provided, adding placeholders")
            while len(parsed_response["suggestions"]) < 5:
                parsed_response["suggestions"].append({
                    "category": "General",
                    "suggestion": "Review and enhance the syllabus to include more specific details.",
                    "impact": "Improves clarity and completeness"
                })

        # Normalize Bloom's Taxonomy levels to ensure English names
        if "bloom_analysis" in parsed_response:
            normalized_found = []
            normalized_missing = []
            for level in parsed_response["bloom_analysis"].get("found_levels", []):
                normalized_level = french_to_english_bloom.get(level, level)
                if normalized_level in blooms_taxonomy:
                    normalized_found.append(normalized_level)
            for level in parsed_response["bloom_analysis"].get("missing_levels", []):
                normalized_level = french_to_english_bloom.get(level, level)
                if normalized_level in blooms_taxonomy:
                    normalized_missing.append(normalized_level)
            parsed_response["bloom_analysis"] = {
                "found_levels": normalized_found,
                "missing_levels": normalized_missing
            }

        return parsed_response
    except Exception as e:
        logger.error(f"Error in enhance_curriculum: {str(e)}")
        raise