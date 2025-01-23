import os
import sqlite3
from concurrent.futures import ThreadPoolExecutor, as_completed

from langchain_core.documents import Document
from tqdm import tqdm
from analyzer_js import analyze_directory
from langchain_chroma import Chroma
from utils import get_llm_query_result, get_ollama_embeddings, get_store_dir_from_repository, \
    store_call_analysis_results, is_binary_file

SUMMARY_SYSTEM_PROMPT = \
    """
    You are an expert software engineer who has been asked to generate a summary of a Python code file.
    Be as concise as possible and do not repeat yourself.
    
    The file is called: {file_name}
    
    The file content is as follows:
    {file_content}
    """

SUMMARY_SYSTEM_PROMPT_CHUNKED = \
    """
    You are an expert software engineer who has been asked to generate a summary of a Python code file.
    Be as concise as possible and do not repeat yourself.
    
    The file is called: {file_name}
    
    A chunk of the file content is as follows:
    {file_content}
    """

SUMMARY_SYSTEM_PROMPT_COMBINE = \
    """
    You are an expert software engineer who has been asked to generate a summary of a Python code file.
    Be as concise as possible and do not repeat yourself.
    
    The file is called: {file_name}
    
    Summaries of chunks of the file:
    {file_content}
    """

SUMMARY_TEMPLATE = \
    """
    Given a Python file, generate a summary of this file to map a given use case requirements to the given python code. 
    The summary should capture the purpose of this file such that given a use case requirement, it can be determined if this class is relevant.
    The summary should be concise not more than 2-3 lines which contains Java Code Keywords present in the class that can be useful to map a usecase requirement to this java code.
    """


def generate_single_file_summary(file):
    result = {
        "file": file['file'],
        "content": "",
        "summary": "",
        "calls": file['calls'],
        "called_by": file['called_by']
    }

    with open(file['file'], "r") as f:
        try:
            if is_binary_file(file['file']):
                return result
            result['content'] = f.read()
            if len(result['content']) > 100000:
                chunks = split_into_chunks(result['content'], 100000, 1000)
                chunked_summaries = [get_llm_query_result(
                    SUMMARY_SYSTEM_PROMPT_CHUNKED.format(file_name=file['file'], file_content=chunk)) for chunk in
                    chunks]
                result['summary'] = get_llm_query_result(
                    SUMMARY_SYSTEM_PROMPT_COMBINE.format(file_name=file['file'], file_content=chunked_summaries))
            else:
                result['summary'] = get_llm_query_result(
                    SUMMARY_SYSTEM_PROMPT.format(file_name=file['file'], file_content=result['content']))
        except UnicodeDecodeError:
            result['content'] = ''
            result['summary'] = ''
    return result


def split_into_chunks(text, chunk_size, overlap):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks


def add_file_contents(file_list, directory):
    store_dir = get_store_dir_from_repository(directory)
    conn = sqlite3.connect(f"{store_dir}/summaries.db")
    cursor = conn.cursor()

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS summaries (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        file TEXT UNIQUE NOT NULL,
        summary TEXT NOT NULL
    )
    """)

    conn.commit()

    results = []
    with ThreadPoolExecutor(max_workers=10) as executor:
        future_to_file = {executor.submit(generate_single_file_summary, file): file for file in file_list}
        for future in tqdm(as_completed(future_to_file), total=len(file_list)):
            try:
                results.append(future.result())
            except Exception as e:
                print(f"Error processing file: {e}")

    for result in results:
        try:
            cursor.execute(
                "INSERT OR IGNORE INTO summaries (file, summary) VALUES (?, ?)",
                (result["file"], result["summary"])
            )
        except sqlite3.Error as e:
            print(f"Error inserting {result['file']}: {e}")

    conn.commit()
    return results


def initialize_vector_dbs(file_list, directory):
    embeddings = get_ollama_embeddings()
    CHUNK_SIZE = 10

    store_dir = get_store_dir_from_repository(directory)

    persist_summary_store_dir = f"{store_dir}/summary_store"
    vector_store_summaries = Chroma(embedding_function=embeddings, persist_directory=persist_summary_store_dir)

    summary_documents = []
    for file in file_list:
        if file['summary'] == '':
            continue
        document = Document(page_content=file['summary'], metadata={"file": file['file']})
        summary_documents.append(document)

    summary_chunks = [summary_documents[i:i + CHUNK_SIZE] for i in range(0, len(summary_documents), CHUNK_SIZE)]
    for chunk in tqdm(summary_chunks):
        vector_store_summaries.add_documents(chunk)

    persist_contents_store_dir = f"{store_dir}/contents_store"
    vector_store_contents = Chroma(embedding_function=embeddings, persist_directory=persist_contents_store_dir)

    content_documents = []
    for file in file_list:
        if file['content'] == '':
            continue
        document = Document(page_content=f"Filename: {file['file']} Content: {file['content']}",
                            metadata={"file": file['file']})
        content_documents.append(document)

    content_chunks = [content_documents[i:i + CHUNK_SIZE] for i in range(0, len(content_documents), CHUNK_SIZE)]
    for chunk in tqdm(content_chunks):
        vector_store_contents.add_documents(chunk)

    return vector_store_summaries, vector_store_contents


def main():
    directory = "/Users/lucas/Downloads/crawlee-python-master"
    directory = "/Users/lucas/Downloads/jitsi-meet-master/react/features/analytics"
    directory = "/Users/lucas/Downloads/jitsi-meet-master/react/features/base/media/components"

    if not os.path.isdir(directory):
        print("Invalid directory path.")
        return

    print("Analyzing directory...")
    file_list = analyze_directory(directory)
    print("Analyzing directory done.")
    print("Storing analysis results...")
    store_call_analysis_results(directory, file_list)
    print("Storing analysis results done.")
    print("Adding file contents and generating summaries...")
    file_list = add_file_contents(file_list, directory)
    print("Adding file contents and generating summaries done.")
    print("Initializing vector databases...")
    initialize_vector_dbs(file_list, directory)
    print("Initializing vector databases done.")


if __name__ == "__main__":
    main()
