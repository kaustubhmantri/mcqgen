import os
import json
import traceback
import pandas as pd
import sentencepiece
from dotenv import load_dotenv
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import streamlit as st
from src.mcqgenerator.utils import read_file, get_table_data
from src.mcqgenerator.logger import logging

# Load environment variables from the .env file
load_dotenv()

# Hugging Face model setup
model_name = "google/flan-t5-large"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

def generate_text(prompt):
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
    outputs = model.generate(**inputs)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Define max_length for model input
MAX_LENGTH = 512

# Quiz Generation Template
template = """
Text: {text}
You are an expert MCQ maker. Given the above text, it is your job to \
create a quiz of {number} multiple choice questions for {subject} students in {tone} tone. 
Make sure the questions are not repeated and check all the questions to be conforming to the text as well.
Make sure to format your response like RESPONSE_JSON below and use it as a guide. \
Ensure to make {number} MCQs.
### RESPONSE_JSON
{response_json}
"""

# Function to truncate text if it exceeds the maximum length
def truncate_text(text):
    if len(text) > MAX_LENGTH:
        return text[:MAX_LENGTH]
    return text

# Function to Generate Quiz
def generate_quiz(text, number, subject, tone, response_json):
    prompt = template.format(text=truncate_text(text), number=number, subject=subject, tone=tone, response_json=response_json)
    result = generate_text(prompt)
    return result

# Quiz Evaluation Template
template2 = """
You are an expert English grammarian and writer. Given a Multiple Choice Quiz for {subject} students,\
you need to evaluate the complexity of the questions and give a complete analysis of the quiz. Only use at max 50 words for complexity analysis. 
If the quiz is not suitable for the cognitive and analytical abilities of the students,\
update the quiz questions which need to be changed and change the tone such that it perfectly fits the student abilities.
Quiz_MCQs:
{quiz}
"""

# Function to Review Quiz
def review_quiz(quiz, subject):
    prompt = template2.format(quiz=truncate_text(quiz), subject=subject)
    result = generate_text(prompt)
    return result

# Overall Function for Generating and Reviewing the Quiz
def generate_and_review_quiz(text, number, subject, tone, response_json):
    quiz = generate_quiz(text, number, subject, tone, response_json)
    review = review_quiz(quiz, subject)
    return quiz, review

# Example usage
text = """
Photosynthesis is the process by which green plants and some other organisms use sunlight to synthesize foods with the help of chlorophyll. 
The process generally involves carbon dioxide and water being converted into glucose and oxygen. Photosynthesis typically occurs in the leaves of plants, specifically in chloroplasts.
"""
number = 2
subject = "Science"
tone = "neutral"
response_json = "{}"

quiz, review = generate_and_review_quiz(text, number, subject, tone, response_json)
print("Quiz:", quiz)
print("Review:", review)
