import os
import PyPDF2
import json
import traceback
import pandas as pd

def read_file(file):
    try:
        if file.name.endswith(".pdf"):
            # Handling PDF file reading
            pdf_reader = PyPDF2.PdfFileReader(file)
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text()
            return text
        elif file.name.endswith(".txt"):
            # Handling Text file reading
            return file.read().decode("utf-8")
        else:
            raise Exception("Unsupported file format. Only PDF and text files are supported.")
    except Exception as e:
        print(f"Error reading the file: {e}")
        raise

def get_table_data(quiz_str):
    try:
        if not quiz_str.strip():
            raise ValueError("Quiz string is empty or contains only whitespace.")
        
        # Load JSON data and parse it
        quiz_dict = json.loads(quiz_str)
        
        # Convert the dictionary into a list of rows for the DataFrame
        table_data = []
        for key, value in quiz_dict.items():
            row = {'question_id': key, **value}
            table_data.append(row)
        
        # Convert to DataFrame
        df = pd.DataFrame(table_data)
        return df

    except json.JSONDecodeError as e:
        print(f"Error decoding JSON: {e}")
        return None
    except Exception as e:
        print(f"An error occurred: {e}")
        return None
