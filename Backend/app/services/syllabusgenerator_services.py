import os
import fitz  # PyMuPDF
import requests

class SyllabusGeneratorService:  
    def __init__(self, ilos_key, topics_key,description_key):
        self.ilos_api_key = ilos_key
        self.topics_api_key = topics_key
        self.description_api_key = description_key
        self.api_url = "https://api.groq.com/openai/v1/chat/completions"
        self.manual_ilos = [
                "Explain how RL differs from other ML paradigms.",
                "Describe fundamental RL concepts.",
                "Formulate decision-making problems as Markov Decision Processes.",
                "Implement RL algorithms including Dynamic Programming, Monte Carlo methods, Temporal Difference learning, Q-learning, and Deep Q Networks (DQN)."
            ]


    def extract_text_from_pdf(self, pdf_path):
        """Extract text from a single PDF file using PyMuPDF (fitz)."""
        try:
            doc = fitz.open(pdf_path)
            text = ''
            for page in doc:
                text += page.get_text()
            return text
        except Exception as e:
            print(f"Failed to read {pdf_path}: {e}")
            return ""
    
    def get_topics_from_groq(self, text):
        """Send extracted text to Groq and retrieve 3 key topics."""
        headers = {
            "Authorization": f"Bearer {self.topics_api_key}",
            "Content-Type": "application/json"
        }

        prompt = (
           
        )

        payload = {
            "model": "",
            "messages": [
                {"role": "system", "content": ""},
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.2,
            "max_tokens": 300
        }

        response = requests.post(self.api_url, headers=headers, json=payload)

        if response.status_code == 200:
            result = response.json()
            return result["choices"][0]["message"]["content"].strip()
        else:
            print(f"Error {response.status_code}: {response.text}")
            return "Error retrieving topics"

    def generate_topics_from_pdfs(self, directory):
        """Extract topics from all PDFs in the given directory."""
        formatted_results = []  # Initialize an empty list to store formatted results
        for filename in sorted(os.listdir(directory)):
            if filename.endswith(".pdf"):  # Only process PDF files
                print(f"Processing: {filename}")
                pdf_path = os.path.join(directory, filename)  # Construct the full file path
                try:
                    text = self.extract_text_from_pdf(pdf_path)  # Extract text from the PDF
                    topics = self.get_topics_from_groq(text)  # Get topics using Groq API
                    
                    formatted = f"--- {filename} ---\n\n{topics}"
                    formatted_results.append(formatted)  # Add formatted result to the list
                except Exception as e:
                    print(f"Error processing {filename}: {e}")  # Handle any errors and log them

        return formatted_results  # Return the list of formatted results
    
    
    

    def generate_ilos_from_groq(self, combined_text):
        headers = {
            "Authorization": f"Bearer {self.ilos_api_key}",
            "Content-Type": "application/json"
        }

        prompt = (
            
        )

        payload = {
            "model": "",
            "messages": [
                {"role": "system", "content": ""},
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.3,
            "max_tokens": 300
        }

        response = requests.post(self.api_url, headers=headers, json=payload)

        if response.status_code == 200:
            result = response.json()
            return result["choices"][0]["message"]["content"].strip()
        else:
            print(f"Error {response.status_code}: {response.text}")
            return "Error retrieving ILOs"

    def generate_combined_ilos(self, directory):
        combined_text = ""
        for filename in sorted(os.listdir(directory)):
            if filename.endswith(".pdf"):
                print(f"Processing: {filename}")
                pdf_path = os.path.join(directory, filename)
                try:
                    chapter_text = self.extract_text_from_pdf(pdf_path)
                    combined_text += chapter_text + "\n"
                except Exception as e:
                    print(f"Error processing {filename}: {e}")
        return self.generate_ilos_from_groq(combined_text)



    def generate_ilos_from_pdfs(self, directory):
        formatted_results = []  # Initialize an empty list to store formatted results
        for filename in sorted(os.listdir(directory)):
            if filename.endswith(".pdf"):  # Only process PDF files
                print(f"Processing: {filename}")
                pdf_path = os.path.join(directory, filename)  # Construct the full file path
                try:
                    text = self.extract_text_from_pdf(pdf_path)  # Extract text from the PDF
                    ilos = self.generate_ilos_from_groq(text)  # Get ILOs using Groq API

                    formatted = f"--- {filename} ---\nHere are the 6 Intended Learning Outcomes (ILOs) for this chapter:\n\n{ilos}"
                    formatted_results.append(formatted)  # Add formatted result to the list
                except Exception as e:
                    print(f"Error processing {filename}: {e}")  # Handle any errors and log them

        return formatted_results  # Return the list of formatted results


    def generate_module_description(self, ilos):
        """
        Generates a concise academic module description (max 2 sentences) based on a list of ILOs.
        """
        ilos_formatted = "\n".join(f"- {ilo}" for ilo in ilos)

        prompt = f"""
           
        """

        headers = {
            "Authorization": f"Bearer {self.description_api_key}",
            "Content-Type": "application/json"
        }

        payload = {
            
        }

        response = requests.post(self.api_url, headers=headers, json=payload)

        if response.status_code == 200:
            result = response.json()
            return result['choices'][0]['message']['content'].strip()
        else:
            raise Exception(f"Groq API request failed: {response.status_code} - {response.text}")

    def generate_manual_description(self):
        """
        Generates a course description using the manually defined ILOs.
        """
        return self.generate_module_description(self.manual_ilos)