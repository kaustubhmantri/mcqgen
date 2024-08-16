import os
import json
import traceback
import pandas as pd
import sys
import sentencepiece
from dotenv import load_dotenv
from transformers import T5Tokenizer, T5ForConditionalGeneration
import streamlit as st
from src.mcqgenerator.utils import read_file, get_table_data
from src.mcqgenerator.logger import logging

# Add your source path
sys.path.append("C:/Users/kaust/mcqgen/src")

# Load environment variables from the .env file
load_dotenv()

# Hugging Face model setup
model_name = "google/flan-t5-large"
tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name)

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

# Streamlit app code
# Creating a title for the app
st.title("MCQs Creator Application with Hugging Face")

# Create a form using st.form
with st.form("user_inputs"):
    # File Upload
    uploaded_file = st.file_uploader("Upload a PDF or txt file")

    # Input Fields
    mcq_count = st.number_input("No of MCQs", min_value=3, max_value=58)

    # Subject
    subject = st.text_input("Insert Subject", max_chars=20)

    # Quiz Tone
    tone = st.text_input("Complexity Level of Questions", max_chars=28, placeholder="Simple")

    # Add Button
    button = st.form_submit_button("Create MCQs")

# Check if the button is clicked and all fields have input
if button and uploaded_file is not None and mcq_count and subject and tone:
    with st.spinner("Generating MCQs..."):
        try:
            # Read the file
            text = read_file(uploaded_file)
            
            # Generate and review quiz
            quiz, review = generate_and_review_quiz(
                text=text,
                number=mcq_count,
                subject=subject,
                tone=tone,
                response_json="{}"  # Pass an empty JSON string if needed
            )

            # Convert the generated quiz into a table format
            if quiz is not None:
                table_data = get_table_data(quiz)
                if table_data is not None:
                    df = pd.DataFrame(table_data)
                    df.index = df.index + 1
                    st.table(df)
                    
                    # Display the review in a text box as well
                    st.text_area(label="Review", value=review, height=200)
                else:
                    st.error("Error processing the table data.")
            else:
                st.error("Quiz generation failed.")

        except Exception as e:
            traceback.print_exception(type(e), e, e.__traceback__)
            st.error("An error occurred during quiz generation.")
