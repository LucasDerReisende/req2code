import mimetypes
import os
import sqlite3
from os import mkdir

import ollama
from dotenv import dotenv_values
from langchain_ollama import OllamaEmbeddings
from openai import OpenAI


def is_binary_file(filename):
    """
    Checks if a given filename corresponds to a file with binary content.

    Parameters:
        filename (str): The path to the file to check.

    Returns:
        bool: True if the file is binary, False otherwise.
    """
    if filename.endswith('.ts'):
        return False

    # Check if file exists
    if not os.path.isfile(filename):
        raise FileNotFoundError(f"The file '{filename}' does not exist.")

    # Guess the file type using mimetypes
    mime_type, encoding = mimetypes.guess_type(filename)
    if mime_type is not None:
        # Check if the MIME type suggests binary content
        if any(mime_type.startswith(prefix) for prefix in ['image/', 'video/', 'audio/', 'application/']):
            return True

    # Read the file to look for binary content
    try:
        with open(filename, 'rb') as file:
            # Read a small portion of the file
            sample = file.read(1024)
            # Check if the sample contains non-text bytes
            if b'\0' in sample:
                return True
    except Exception as e:
        print(f"An error occurred while reading the file: {e}")
        return False

    return False

def store_call_analysis_results(repo_dir, files):
    store_dir = get_store_dir_from_repository(repo_dir)
    conn = sqlite3.connect(f"{store_dir}/call_analysis.db")
    cursor = conn.cursor()

    cursor.execute('''
    CREATE TABLE IF NOT EXISTS files (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        file_name TEXT NOT NULL UNIQUE
    );
    ''')

    cursor.execute('''
    CREATE TABLE IF NOT EXISTS file_relations (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        caller_id INTEGER NOT NULL,
        called_id INTEGER NOT NULL,
        FOREIGN KEY (caller_id) REFERENCES files(id) ON DELETE CASCADE,
        FOREIGN KEY (called_id) REFERENCES files(id) ON DELETE CASCADE
    );
    ''')
    conn.commit()

    for file in files:
        # Insert file if it doesn't exist
        cursor.execute('INSERT OR IGNORE INTO files (file_name) VALUES (?);', (file['file'],))

        # Get the file ID
        cursor.execute('SELECT id FROM files WHERE file_name = ?;', (file['file'],))
        file_id = cursor.fetchone()[0]

        # Insert "calls" relations
        for called_file in file['calls']:
            cursor.execute('INSERT OR IGNORE INTO files (file_name) VALUES (?);', (called_file,))

            cursor.execute('SELECT id FROM files WHERE file_name = ?;', (called_file,))
            called_id = cursor.fetchone()[0]

            cursor.execute('INSERT INTO file_relations (caller_id, called_id) VALUES (?, ?);', (file_id, called_id))

        # Insert "called_by" relations
        for caller_file in file['called_by']:
            cursor.execute('INSERT OR IGNORE INTO files (file_name) VALUES (?);', (caller_file,))

            cursor.execute('SELECT id FROM files WHERE file_name = ?;', (caller_file,))
            caller_id = cursor.fetchone()[0]

            cursor.execute('INSERT INTO file_relations (caller_id, called_id) VALUES (?, ?);', (caller_id, file_id))

    conn.commit()

def load_call_analysis_results(repo_dir):
    store_dir = get_store_dir_from_repository(repo_dir)
    conn = sqlite3.connect(f"{store_dir}/call_analysis.db")
    cursor = conn.cursor()

    # Get all files
    cursor.execute('SELECT id, file_name FROM files;')
    files = {row[0]: row[1] for row in cursor.fetchall()}

    # Initialize result list
    result = []

    for file_id, file_name in files.items():
        # Get files called by this file
        cursor.execute('''
            SELECT f2.file_name FROM file_relations
            JOIN files f2 ON file_relations.called_id = f2.id
            WHERE file_relations.caller_id = ?;
            ''', (file_id,))
        calls = [row[0] for row in cursor.fetchall()]

        # Get files calling this file
        cursor.execute('''
            SELECT f1.file_name FROM file_relations
            JOIN files f1 ON file_relations.caller_id = f1.id
            WHERE file_relations.called_id = ?;
            ''', (file_id,))
        called_by = [row[0] for row in cursor.fetchall()]

        result.append({"file": file_name, "calls": calls, "called_by": called_by})

    return result


def get_openai_client():
    client = OpenAI(
        api_key=dotenv_values(".env")["OPENAI_API_KEY"]
    )

    return client

def get_llm_query_result(query):
    return get_openai_query_result(query)

def get_openai_query_result(query):
    client = get_openai_client()
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "user",
                "content": query,
            }
        ]
    )
    return response.choices[0].message.content

def get_local_model():
    return "llama3.1:8B"

def get_local_llm_query_result(query):
    response = ollama.chat(
        model=get_local_model(),
        messages=[
            {
                "role": "user",
                "content": query,
            }
        ]
    )
    return response["message"]["content"]

def get_ollama_embeddings():
    embeddings = OllamaEmbeddings(model=get_local_model())
    return embeddings

def get_store_dir_from_repository(repository_path):
    store_dir = f"{repository_path}/stores"
    if not os.path.exists(store_dir):
        mkdir(store_dir)
    return store_dir