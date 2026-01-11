"""
Configuration file for the Spam Detection Project
"""

import os
from dotenv import load_dotenv

load_dotenv()


GROQ_API_KEY = os.getenv("GROQ_API_KEY", "your-groq-api-key-here")


MODEL_CONFIG = {
    "temperature": 0.2,
    "max_tokens": 2048
}


NAIVE_BAYES_CONFIG = {
    "smoothing": 1.0,  
    "min_word_length": 2
}


PROJECT_SETTINGS = {
    "test_split": 0.2,
    "random_seed": 42
}