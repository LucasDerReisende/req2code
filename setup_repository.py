import os
from concurrent.futures import ThreadPoolExecutor, as_completed

from langchain_core.documents import Document
from tiktoken import get_encoding
from tqdm import tqdm
from langchain_chroma import Chroma
from utils import get_llm_query_result, get_embeddings, get_store_dir_from_repository, load_call_analysis_results, \
    load_summaries, \
    store_call_analysis_results, is_binary_file, store_summaries, get_initial_files, join_file_lists, \
    get_embedding_tokens, set_embedding_tokens, get_model_encoding_string

CALC_EMBEDDING_TOKENS = True

SUMMARY_SYSTEM_PROMPT_CHUNKED = \
    """
    You are an expert software engineer who has been asked to generate a summary of a code file.
    The summary should not only contain information about the code, but also what it is used for, its purpose, including keywords at the end.
    The summary will later be stored in a vector database for retrieval.
    It will be compared to software requirements to determine if it is relevant, so make sure to include relevant information.
    Be as concise as possible and do not repeat yourself.
    
    The file is called: {file_name}
    
    A chunk of the file content is as follows:
    {file_content}
    """


def generate_single_file_summaries(directory, file):
    result = {
        "file": file['file'],
        "content": "",
        "summaries": [],
        "calls": file['calls'],
        "called_by": file['called_by']
    }

    file_path = os.path.join(directory, file['file'])
    with open(file_path, "r") as f:
        try:
            if is_binary_file(file_path):
                return result

            result['content'] = f.read()
            chunks = split_into_chunks(result['content'], 100000, 1000)
            chunked_summaries = [
                get_llm_query_result(SUMMARY_SYSTEM_PROMPT_CHUNKED.format(file_name=file['file'], file_content=chunk))
                for chunk in chunks
            ]
            result['summaries'] = chunked_summaries
        except UnicodeDecodeError:
            result['content'] = ''
            result['summaries'] = []
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
    results = []
    with ThreadPoolExecutor(max_workers=10) as executor:
        future_to_file = {executor.submit(generate_single_file_summaries, directory, file): file for file in file_list}
        for future in tqdm(as_completed(future_to_file), total=len(file_list)):
            try:
                results.append(future.result())
            except Exception as e:
                print(f"Error processing file: {e}")

    store_summaries(results, directory)
    return results


def initialize_summary_vector_db(file_list, directory):
    store_dir = get_store_dir_from_repository(directory)

    embeddings = get_embeddings()
    CHUNK_SIZE = 10

    # Initialize the summary vector store
    persist_summary_store_dir = f"{store_dir}/summary_store"
    vector_store_summaries = Chroma(embedding_function=embeddings, persist_directory=persist_summary_store_dir)

    if CALC_EMBEDDING_TOKENS:
        encoding = get_encoding(get_model_encoding_string())

    # Prepare summary documents for vectorization
    summary_documents = []
    for file in file_list:
        if not file['summaries']:
            continue
        for summary in file['summaries']:
            document = Document(page_content=summary, metadata={"file": file['file']})
            summary_documents.append(document)
            if CALC_EMBEDDING_TOKENS:
                set_embedding_tokens(get_embedding_tokens() + len(encoding.encode(document.page_content)))

    # Chunk the summary documents and add them to the vector store
    summary_chunks = [summary_documents[i:i + CHUNK_SIZE] for i in range(0, len(summary_documents), CHUNK_SIZE)]
    for chunk in tqdm(summary_chunks):
        vector_store_summaries.add_documents(chunk)

    return vector_store_summaries


def initialize_content_vector_db(file_list, directory):
    store_dir = get_store_dir_from_repository(directory)

    embeddings = get_embeddings()
    CHUNK_SIZE = 10

    # Initialize the content vector store
    persist_contents_store_dir = f"{store_dir}/contents_store"
    vector_store_contents = Chroma(embedding_function=embeddings, persist_directory=persist_contents_store_dir)

    if CALC_EMBEDDING_TOKENS:
        encoding = get_encoding(get_model_encoding_string())

    # Prepare content documents for vectorization
    content_documents = []
    for file in file_list:
        if file['content'] == '':
            continue
        document = Document(page_content=f"Filename: {file['file']} Content: {file['content']}",
                            metadata={"file": file['file']})
        content_documents.append(document)
        if CALC_EMBEDDING_TOKENS:
            set_embedding_tokens(get_embedding_tokens() + len(encoding.encode(document.page_content)))

    # Chunk the content documents and add them to the vector store
    content_chunks = [content_documents[i:i + CHUNK_SIZE] for i in range(0, len(content_documents), CHUNK_SIZE)]
    for chunk in tqdm(content_chunks):
        vector_store_contents.add_documents(chunk)

    # Return the initialized vector stores
    return vector_store_contents


def init_project(directory, analyze_fn, args):
    if not any([args.analyse, args.summarize, args.vectorize_content, args.vectorize_summaries]):
        print(
            "choose at least one of the following options: --analyse, --summarize, --vectorize-content, --vectorize-summaries")
        return

    all_files = get_initial_files(directory)

    if args.analyse:
        print("Analyzing directory...")
        if analyze_fn is None:
            print("Analysis function not specified for this project.")
            return
        import_graph = analyze_fn(directory)
        print("Analyzing directory done.")
        print("Storing analysis results...")
        store_call_analysis_results(directory, import_graph)
        print("Storing analysis results done.")

    if args.summarize:
        file_list = load_call_analysis_results(directory)
        file_list = join_file_lists(file_list, all_files)
        print("Adding file contents and generating summaries...")
        file_list = add_file_contents(file_list, directory)
        print("Adding file contents and generating summaries done.")

    if args.vectorize_content or args.vectorize_summaries:
        analysis_results = {descr['file']: descr for descr in load_call_analysis_results(directory)}
        summary_results = {descr['file']: descr for descr in load_summaries(directory)}
        file_list = [{**analysis_results.get(id, {'calls': [], 'called_by': []}), **summaries} for id, summaries in
                     summary_results.items()]

    if args.vectorize_summaries:
        print("Initializing summary vector database...")
        initialize_summary_vector_db(file_list, directory)
        print("Initializing summary vector database done.")
    if args.vectorize_content:
        print("Initializing content vector database...")
        initialize_content_vector_db(file_list, directory)
        print("Initializing content vector database done.")
