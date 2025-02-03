import json
import sqlite3
import os
from langchain_chroma import Chroma

from utils import get_embeddings, get_store_dir_from_repository, get_llm_query_result, load_call_analysis_results, \
    get_file_summaries_dict


def similar_files_vector_db(query, directory):
    embeddings = get_embeddings()
    store_dir = get_store_dir_from_repository(directory)

    persist_summary_store_dir = f"{store_dir}/summary_store"
    vector_store_summaries = Chroma(embedding_function=embeddings, persist_directory=persist_summary_store_dir)

    persist_contents_store_dir = f"{store_dir}/contents_store"
    vector_store_contents = Chroma(embedding_function=embeddings, persist_directory=persist_contents_store_dir)

    similar_documents_summaries = vector_store_summaries.similarity_search(query, k=10)
    similar_documents_contents = vector_store_contents.similarity_search(query, k=10)

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
    store_dir = get_store_dir_from_repository(directory)

    def get_size():
        total_size = 0
        for dirpath, dirnames, filenames in os.walk(store_dir):
            for f in filenames:
                fp = os.path.join(dirpath, f)
                # skip if it is symbolic link
                if not os.path.islink(fp):
                    total_size += os.path.getsize(fp)

        return total_size

    print(f'disk size of databases: {get_size() / 1024 / 1024:.2f} MB')
    # ..to be continued


def reformulate_query_for_retrieval(query):
    TEMPLATE = """
    In the following you will be given a requirement for a software project.
    We setup a vector database with embeddings of summaries and contents of the files in the project.
    We will use this database to find similar files to the requirement.
    Please rewrite the requirement such that the similarity search will be most successful for finding the relevant files.
    Return ONLY the rewritten query string for the similarity search.
    
    {query}
    """

    TEMPLATE = """
    In the following you will be given a requirement for a software project.
    We setup a vector database with embeddings of summaries and contents of the files in the project.
    We will use this database to find similar files to the requirement.
    Please return keywords such that the similarity search will be most successful for finding the relevant files.
    Return ONLY the keywords for the similarity search.

    {query}
    """

    return get_llm_query_result(TEMPLATE.format(query=query))


def filter_similar_files_by_summary(query, similar_files, directory):
    TEMPLATE = """
    You are given a requirement for a software project and a list of files that are similar to the requirement.
    For each file, you are given a summary of the file content.
    Please filter the list of files to remove files that do not have anything to do with the requirement.
    Do not hallucinate file names!
    Return only a list of the relevant files in JSON format. [<"File 1 Name">, <"File 2 Name">, ... <"File N Name">]
       
    Requirement:
    {requirement}
    
    Similar files:
    {files}
    """

    summaries = get_file_summaries_dict(directory, similar_files)

    query = TEMPLATE.format(requirement=query, files="\n\n".join([f"{file}:\n {summary}" for file, summary in summaries.items()]))

    result = get_llm_query_result(query)
    try:
        result = json.loads(result.replace('```json\n', '').replace('```', ''))
    except Exception as e:
        print(f"Error parsing result: {e}")
        return []
    return result

def find_missing_files(query, similar_files, directory):
    TEMPLATE = """
    You are given a requirement for a software project and a list of files with their summaries that are similar to the requirement.
    Which other files are still needed to fulfill the requirement?
    Generate a string that can be used to search a vector database for files that are missing.
    This vector database contains embeddings of summaries and contents of the files in the project.
    Return ONLY the string for the search.
    
    Requirement:
    {requirement}
    
    Similar files:
    {files}
    """

    summaries = get_file_summaries_dict(directory, similar_files)

    search_string = get_llm_query_result(TEMPLATE.format(requirement=query, files="\n\n".join([f"{file}:\n {summary}" for file, summary in summaries.items()])))

    return similar_files_vector_db(search_string, directory)

def get_summary(query, similar_files, directory):
    TEMPLATE = """
    You are given a requirement for a software project and a list of files with their summaries that are similar to the requirement.
    A user needs to implement this requirement and through preprocessing we selected a list of files that could be relevant.
    Please provide a summary in the following format to the user:
    
    **Summary**
    *Summary text for the changes to be performed*
    
    **Step-by-step Breakdown**
    *Step-by-step breakdown of the changes to be performed*
    *Structure the steps by components, so the user knows which need to be changed*
    *Example*
    a. Image component
     - Change the image source to the new image
    b. Backend
     - Add a new endpoint for the new feature
    *End of example*
    Do not give or generate code samples!
    Do not hallucinate steps that are not related to the relevant files!
    Be very concise in the steps!
    Do not make up steps that are unrelated to the given files!
    
    **Files to be changed**
    *List of files that need to be changed in a markdown list*
    *Do not give additional files that are not related to the requirement or hallucinate files*
    *Only write file names that were given before*
    
    **Estimation**
    *Estimation of the time needed to implement the changes*
    *First write a one sentence summary of the estimation*
    *Then write: Estimation: X hours*
    
    Requirement:
    {requirement}
    
    Similar files:
    {files}
    """

    summaries = get_file_summaries_dict(directory, similar_files)

    query = TEMPLATE.format(requirement=query, files="\n\n".join([f"{file}:\n {summary}" for file, summary in summaries.items()]))
    return get_llm_query_result(query)


def query_project(directory, args):
    # find similar files
    print("Generating similarity query...")
    reformulated_query = reformulate_query_for_retrieval(args.query)
    print("Generating similarity query done")
    print('Finding similar files...')
    similar_files = set(similar_files_vector_db(reformulated_query, directory))
    print('Finding similar files done')


    if args.adjacent:
        # find and add adjacent files
        print('Finding adjacent files...')
        file_list = load_call_analysis_results(directory)
        adjacent_files = set()
        for file in file_list:
            if file['file'] in similar_files:
                for called_file in file['calls']:
                    adjacent_files.add(called_file)
                for calling_file in file['called_by']:
                    adjacent_files.add(calling_file)
        similar_files = similar_files.union(adjacent_files)
        print('Finding adjacent files done')

    if args.find_missing:
        print('Finding missing files...')
        missing_files = find_missing_files(args.query, list(similar_files), directory)
        similar_files = similar_files.union(missing_files)
        print('Finding missing files done')

    if args.filter_files:
        print('Filtering similar files...')
        similar_files = filter_similar_files_by_summary(args.query, list(similar_files), directory)
        print('Filtering similar files done')

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
        return

    print('Generating summary...')
    summary = get_summary(args.query, result, directory)
    print('Summary:\n', summary)
