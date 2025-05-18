import os
import logging
from groq import Groq, APIConnectionError, AuthenticationError

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

class GroqClient:
    def __init__(self):
        self.client = None
        self.api_key = os.getenv("GROQ_API_KEY")
        if not self.api_key:
            logger.error("GROQ_API_KEY environment variable is not set")
            raise ValueError("GROQ_API_KEY environment variable is not set")
        
        try:
            logger.info("Initializing Groq client")
            self.client = Groq(api_key=self.api_key)
            logger.info("Groq client initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize Groq client: {str(e)}")
            raise ValueError(f"Failed to initialize Groq client: {str(e)}")

    def generate(self, prompt, max_new_tokens=4000, temperature=0.2, top_p=0.9):
        if not self.client:
            logger.error("Groq client not initialized")
            raise ValueError("Groq client not initialized")
        
        try:
            logger.debug(f"Sending prompt to Groq (first 500 chars): {prompt[:500]}...")
            completion = self.client.chat.completions.create(
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt}
                ],
                model="llama3-70b-8192",
                max_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p
            )
            response = completion.choices[0].message.content
            logger.debug(f"Received response from Groq (first 500 chars): {response[:500]}...")
            return response
        except AuthenticationError as e:
            logger.error(f"Authentication error: {str(e)}")
            raise ValueError(f"Invalid Groq API key: {str(e)}")
        except APIConnectionError as e:
            logger.error(f"API connection error: {str(e)}")
            raise ValueError(f"Failed to connect to Groq API: {str(e)}")
        except Exception as e:
            logger.error(f"Error generating response from Groq: {str(e)}")
            raise ValueError(f"Error generating response from Groq: {str(e)}")

# Singleton instance
groq_client = None

def initialize_groq_client():
    global groq_client
    if groq_client is None:
        groq_client = GroqClient()
    return groq_client

def generate_response(prompt, max_new_tokens=4000, temperature=0.2, top_p=0.9):
    try:
        client = initialize_groq_client()
        return client.generate(prompt, max_new_tokens, temperature, top_p)
    except Exception as e:
        logger.error(f"Error in generate_response: {str(e)}")
        raise