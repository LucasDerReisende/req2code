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

total_input_tokens = 0
total_output_tokens = 0
total_embedding_tokens = 0

def get_input_tokens():
    return total_input_tokens

def get_output_tokens():
    return total_output_tokens

def get_embedding_tokens():
    return total_embedding_tokens

def set_input_tokens(value):
    global total_input_tokens
    total_input_tokens = value

def set_output_tokens(value):
    global total_output_tokens
    total_output_tokens = value

def set_embedding_tokens(value):
    global total_embedding_tokens
    total_embedding_tokens = value

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
        if any(mime_type.startswith(prefix) for prefix in ['image/', 'video/', 'audio/', 'application/']) and mime_type != 'application/json':
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

def get_initial_files(directory):
    blacklist = ['node_modules', '\.(.*)$', '__pycache__', '(.*)\.lock', 'package-lock.json']
    file_list = []
    for root, dirs, files in os.walk(directory, topdown=True):
        # Skip directories that match the blacklist
        dirs[:] = [d for d in dirs if re.match('|'.join(blacklist), d) is None]
        files[:] = [f for f in files if re.match('|'.join(blacklist), f) is None]
        for file in files:
            relative_path = os.path.join(os.path.relpath(root, directory), file)
            full_path = os.path.join(root, file)

            result = {
                "file": relative_path,
                "content": "",
                "calls": [],
                "called_by": []
            }
            with open(full_path, "r") as f:
                try:
                    if is_binary_file(full_path):
                        continue

                    result['content'] = f.read()
                except UnicodeDecodeError:
                    pass
            file_list.append(result)
    return file_list

def join_file_lists(files1, files2):
    result = []
    files1_dict = {file['file']: file for file in files1}
    files2_dict = {file['file']: file for file in files2}
    for file_name in set(files1_dict.keys()).union(files2_dict.keys()):
        if file_name in files1_dict:
            result.append(files1_dict[file_name])
        else:
            result.append(files2_dict[file_name])
    return result


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
        for summary in result['summaries']:
            try:
                cursor.execute(
                    "INSERT OR IGNORE INTO summaries (file, content, summary) VALUES (?, ?, ?)",
                    (result["file"], result["content"], summary)
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

    result = {}
    for file, content, summary in summaries:
        if file in result:
            result[file]["summaries"].append(summary)
        else:
            result[file] = {
                "file": file,
                "content": content,
                "summaries": [summary]
            }
    return list(result.values())

def get_file_summaries_dict(directory, files):
    store_dir = get_store_dir_from_repository(directory)
    conn = sqlite3.connect(f"{store_dir}/summaries.db")
    cursor = conn.cursor()

    summaries = {}
    for file in files:
        cursor.execute("SELECT summary FROM summaries WHERE file=?", (file,))
        summary_list = cursor.fetchall()
        if not summary_list:
            continue
        summary_list = [summary[0] for summary in summary_list]
        summaries[file] = summary_list

    cursor.close()
    conn.close()

    return summaries

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
        ],
        temperature=0.1,
    )
    set_input_tokens(get_input_tokens() + response.usage.prompt_tokens)
    set_output_tokens(get_output_tokens() + response.usage.completion_tokens)

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

def get_model_encoding_string():
    return "cl100k_base"

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

def print_runtime(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"Runtime: {end_time - start_time:.2f} seconds")
        return result
    return wrapper