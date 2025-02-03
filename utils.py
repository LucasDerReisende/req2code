import mimetypes
import os
import re
import sqlite3
import time

import ollama
from dotenv import dotenv_values
from langchain_ollama import OllamaEmbeddings
from langchain_openai import OpenAIEmbeddings
from openai import OpenAI, RateLimitError as OpenAIRateLimitError

from tenacity import retry, stop_after_attempt, retry_if_exception_type


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

def store_summaries(files, directory):
    store_dir = get_store_dir_from_repository(directory)
    conn = sqlite3.connect(f"{store_dir}/summaries.db")
    cursor = conn.cursor()

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS summaries (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        file TEXT UNIQUE NOT NULL,
        content TEXT NOT NULL,
        summary TEXT NOT NULL
    )
    """)

    conn.commit()
    
    for result in files:
        try:
            cursor.execute(
                "INSERT OR IGNORE INTO summaries (file, content, summary) VALUES (?, ?, ?)",
                (result["file"], result["content"], result["summary"])
            )
        except sqlite3.Error as e:
            print(f"Error inserting {result['file']}: {e}")

    conn.commit()

def load_summaries(directory):
    store_dir = get_store_dir_from_repository(directory)
    conn = sqlite3.connect(f"{store_dir}/summaries.db")
    cursor = conn.cursor()

    cursor.execute("SELECT file, content, summary FROM summaries")
    
    summaries = cursor.fetchall()
    
    return [{"file": file, "content": content, "summary": summary} for file, content, summary in summaries]

def get_openai_client():
    client = OpenAI(
        api_key=dotenv_values(".env")["OPENAI_API_KEY"]
    )
    return client

def get_deepseek_client():
    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=dotenv_values(".env")["DEEPSEEK_API_KEY"]
    )
    return client

def get_llm_query_result(query):
    return get_openai_query_result(query)


def openai_rate_limit_handler(retry_state):
    exception = retry_state.outcome.exception()
    match = re.search(r"Please try again in ([\d.]+)s", str(exception.message))
    if match:
        wait_time = float(match.group(1))
        print(f"Rate limit hit. Retrying in {wait_time:.2f} seconds.")
        time.sleep(wait_time + 0.2)

class DeepSeekTimeout(Exception):
    def __init__(self, message=0):
        self.message = message
        super().__init__(self.message)

def deepseek_rate_limit_handler(retry_state):
    exception = retry_state.outcome.exception()
    wait_time = int(exception.message) / 1000.0 - time.time()
    print(f"Rate limit hit. Retrying in {wait_time:.2f} seconds.")
    time.sleep(wait_time + 0.2)
    
# Retry logic for API calls, independent per thread
@retry(
    retry=retry_if_exception_type(OpenAIRateLimitError),  # Retry only on RateLimitError
    stop=stop_after_attempt(5),  # Retry up to 5 times
    before_sleep=openai_rate_limit_handler
)
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

@retry(
    retry=retry_if_exception_type(DeepSeekTimeout),  # Retry only on RateLimitError
    stop=stop_after_attempt(5),  # Retry up to 5 times
    before_sleep=deepseek_rate_limit_handler
)
def get_deepseek_query_result(query):
    client = get_deepseek_client()
    response = client.chat.completions.create(
        model="deepseek/deepseek-r1:free",
        messages=[
            {
                "role": "user",
                "content": query,
            }
        ]
    )
    if response.model_extra.get('error').get('code') == 429:
        raise DeepSeekTimeout(response.model_extra.get('error').get('metadata').get('headers').get('X-RateLimit-Reset'))
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

def get_embeddings():
    return get_openai_embeddings()

def get_ollama_embeddings():
    embeddings = OllamaEmbeddings(model=get_local_model())
    return embeddings

def get_openai_embeddings():
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small",
        openai_api_key=dotenv_values(".env")["OPENAI_API_KEY"]
    )
    return embeddings

def get_store_dir_from_repository(repository_path):
    store_dir = os.path.join("./data", os.path.split(repository_path)[1])
    os.makedirs(store_dir, exist_ok=True)
    
    return store_dir