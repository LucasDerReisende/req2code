import argparse
import os
import sqlite3
from concurrent.futures import ThreadPoolExecutor, as_completed

from langchain_core.documents import Document
from tqdm import tqdm
from analyzer_js import analyze_directory as analyze_js_directory
from analyzer_py import analyze_directory as analyze_py_directory
from langchain_chroma import Chroma
from query_requirement import query_project, query_stats
from setup_repository import add_file_contents, init_project, initialize_content_vector_db, initialize_summary_vector_db
from utils import get_llm_query_result, get_ollama_embeddings, get_store_dir_from_repository, load_call_analysis_results, load_summaries, \
    store_call_analysis_results, is_binary_file, store_summaries


def main():
    projects = {
        "crawlee_python_master": ("/Users/lucas/Downloads/crawlee-python-master", analyze_py_directory),
        "jitsi_analytics": ("/Users/lucas/Downloads/jitsi-meet-master/react/features/analytics", analyze_js_directory),
        "jitsi_media": ("/Users/lucas/Downloads/jitsi-meet-master/react/features/base/media/components", analyze_js_directory),
        "newsscraper": ("../../NewsPolitics/newsscraper", analyze_py_directory),
        "cula": ("data/cula", None),
    }
    
    parser = argparse.ArgumentParser()
    parser.add_argument("project", help="The project to analyze", choices=projects.keys())
    subparsers = parser.add_subparsers(dest="command", required=True)
    
    init_parser = subparsers.add_parser("init", help="Initialize the project")
    init_parser.add_argument("--analyse", action="store_true", help="Analyze the directory")
    init_parser.add_argument("--summarize", action="store_true", help="Summarize the contents")
    init_parser.add_argument("--vectorize-summaries", action="store_true", help="Vectorize the summaries")
    init_parser.add_argument("--vectorize-content", action="store_true", help="Vectorize the contents")
    
    query_parser = subparsers.add_parser("retrieve", help="Query the database for similar files")
    query_group = query_parser.add_mutually_exclusive_group(required=True)
    query_group.add_argument("--query", help="The query string")
    query_group.add_argument("--stats", action="store_true", help="Show the stats of the project")    
    
    args = parser.parse_args()
    
    (directory, analyze_fn) = projects[args.project]
    
    # get abs path, in case of relative
    directory = os.path.abspath(directory)    
    if not os.path.isdir(directory):
        print("Invalid directory path.")
        return
    
    if args.command == "init":
        init_project(directory, analyze_fn, args)
        
    if args.command == "retrieve":
        if args.query:
            # requirement = """
            # It would be handy to have the ability to select multiple items in the filter inputs for events. I have a camera that generates pretty constant events, so when I visit events it is pages of "bird bird bird bird" as my wife likes to see what her chickens were up to during the day. All other categories are interesting to me, but it's pages of scrolling to get to it unless I select a different label, but then I have to look at labels one at a time.
            # The ability to select all and then deselect one or more labels -- or simply select multiple labels individually -- would be a nice to have!
            # """
            # requirement = "Bumps vite from 2.8.6 to 2.9.13."
            # requirement = "Refactor events to be more generic"
            # requirement = "Configuration of the camera"
            # requirement = "Refactor the session"
            # requirement = "Refactor all related to Audio"
            # args.query = requirement
            
            query_project(directory, args)
        if args.stats:
            query_stats(directory, args)
        
if __name__ == "__main__":
    main()