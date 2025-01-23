import sqlite3

from langchain_chroma import Chroma

from utils import get_ollama_embeddings, get_store_dir_from_repository, get_llm_query_result, load_call_analysis_results

from analyzer_py import analyze_directory

def similar_files_vector_db(requirement, directory):
    embeddings = get_ollama_embeddings()
    store_dir = get_store_dir_from_repository(directory)

    persist_summary_store_dir = f"{store_dir}/summary_store"
    vector_store_summaries = Chroma(embedding_function=embeddings, persist_directory=persist_summary_store_dir)

    persist_contents_store_dir = f"{store_dir}/contents_store"
    vector_store_contents = Chroma(embedding_function=embeddings, persist_directory=persist_contents_store_dir)

    similar_documents_summaries = vector_store_summaries.similarity_search(requirement, k=10)
    similar_documents_contents = vector_store_contents.similarity_search(requirement, k=5)

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

        with open(file, 'r') as f:
            content = f.read()

        files.append(f"{file}: {content}")


    cursor.close()
    conn.close()

    query = TEMPLATE.format(requirement=requirement, files="\n".join(files))
    return get_llm_query_result(query)

def main():
    directory = "/Users/lucas/Downloads/crawlee-python-master"
    directory = "/Users/lucas/Downloads/jitsi-meet-master/react/features/base/media/components"


    requirement = """
    It would be handy to have the ability to select multiple items in the filter inputs for events. I have a camera that generates pretty constant events, so when I visit events it is pages of "bird bird bird bird" as my wife likes to see what her chickens were up to during the day. All other categories are interesting to me, but it's pages of scrolling to get to it unless I select a different label, but then I have to look at labels one at a time.
    The ability to select all and then deselect one or more labels -- or simply select multiple labels individually -- would be a nice to have!
    """
    requirement = "Bumps vite from 2.8.6 to 2.9.13."
    requirement = "Refactor events to be more generic"
    requirement = "Configuration of the camera"
    requirement = "Refactor the session"
    requirement = "Refactor all related to Audio"

    # find similar files
    print('Finding similar files...')
    similar_files = set(similar_files_vector_db(requirement, directory))
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
    relevant_files = get_relevant_files(requirement, list(similar_files), directory)
    print('Getting relevant files done')

    print('Relevant files:', relevant_files)



if __name__== "__main__":
    main()
