import json
import sqlite3
import os
from langchain_chroma import Chroma

from utils import get_ollama_embeddings, get_store_dir_from_project, get_store_dir_from_repository, get_llm_query_result, load_call_analysis_results

from analyzer_py import analyze_directory

def similar_files_vector_db(query, project):
    embeddings = get_ollama_embeddings()
    store_dir = get_store_dir_from_project(project)

    persist_summary_store_dir = f"{store_dir}/summary_store"
    vector_store_summaries = Chroma(embedding_function=embeddings, persist_directory=persist_summary_store_dir)

    persist_contents_store_dir = f"{store_dir}/contents_store"
    vector_store_contents = Chroma(embedding_function=embeddings, persist_directory=persist_contents_store_dir)

    similar_documents_summaries = vector_store_summaries.similarity_search(query, k=10)
    similar_documents_contents = vector_store_contents.similarity_search(query, k=5)

    similar_files_summaries = [document.metadata["file"] for document in similar_documents_summaries]
    similar_files_contents = [document.metadata["file"] for document in similar_documents_contents]

    return similar_files_summaries + similar_files_contents


def get_relevant_files(requirement, file_list, directory):
    TEMPLATE = \
        """
        What are the names of the files that are related to the following use case requirement?
        {requirement}
    
        Provide the answer in a list format and provide ONLY the list of file names with path as a JSON list.
        [<"File 1 Name">, <"File 2 Name">, ... <"File N Name">]
        ONLY return data in this format! Don't write additional text!
        
        Given are some files with their content for context:
        
        {files}
        """

    store_dir = get_store_dir_from_repository(directory)
    conn = sqlite3.connect(f"{store_dir}/summaries.db")
    cursor = conn.cursor()

    files = []
    for file in file_list:
        cursor.execute("SELECT summary FROM summaries WHERE file=?", (file,))
        # summary = cursor.fetchone()

        with open(os.path.join(directory, file), 'r') as f:
            content = f.read()

        files.append(f"{file}: {content}")


    cursor.close()
    conn.close()

    query = TEMPLATE.format(requirement=requirement, files="\n".join(files))
    return get_llm_query_result(query)

def query_stats(directory, args):
    store_dir = get_store_dir_from_project(args.project)
    # size of store_dir
    print('disk size of databases:', os.path.getsize(store_dir))
    # ..to be continued

def query_project(directory, args):
    # find similar files
    print('Finding similar files...')
    similar_files = set(similar_files_vector_db(args.query, args.project))
    print('Finding similar files done')

    # find and add adjacent files
    print('Finding adjacent files...')
    file_list = load_call_analysis_results(directory)
    for file in file_list:
        if file['file'] in similar_files:
            for called_file in file['calls']:
                similar_files.add(called_file)
            for calling_file in file['called_by']:
                similar_files.add(calling_file)
    print('Finding adjacent files done')

    # get relevant files
    print('Getting relevant files...')
    relevant_files = get_relevant_files(args.query, list(similar_files), directory)
    print('Getting relevant files done')

    print('Relevant files:', relevant_files)
    
    try:
        result = json.loads(relevant_files.replace('```json\n', '').replace('```', ''))
        print('Relevant files:', result)
    except Exception as e:
        print(f"Error parsing relevant files: {e}")
