import os
import ast
from collections import defaultdict
from re import match

def analyze_directory(directory):
    """
    Analyze a directory of Python files to find file call relationships.

    Args:
        directory (str): Path to the directory to analyze.

    Returns:
        list: A list of dictionaries, each representing a file with its calls and called_by relationships.
    """
    file_calls = defaultdict(set)
    file_called_by = defaultdict(set)
    file_list = []
    
    blacklist = ['node_modules', '\.(.*)$', '__pycache__', '(.*)\.lock', 'package-lock.json']
    whitelist = ['(.*)\.py$']

    # Step 1: Map filenames to full paths
    for root, dirs, files in os.walk(directory, topdown=True):
        # Skip directories that match the blacklist
        dirs[:] = [d for d in dirs if match('|'.join(blacklist), d) is None]
        files[:] = [f for f in files if match('|'.join(blacklist), f) is None]
        for file in files:
            relative_path = os.path.join(os.path.relpath(root, directory), file)
            file_list.append(relative_path)
            full_path = os.path.join(root, file)

    # Helper to resolve module to files
    def resolve_module_to_files(base_path, module_name):
        module_path = module_name.replace(".", os.sep)
        
        # Check relative to the directory of the base_path
        base_dir = os.path.dirname(os.path.join(directory, base_path))
        possible_file_path = os.path.join(base_dir, module_path + ".py")
        possible_dir_path = os.path.join(base_dir, module_path)

        resolved_files = []

        # Case 1: Check if it's a .py file
        if os.path.isfile(possible_file_path):
            resolved_files.append(os.path.relpath(possible_file_path, directory))

        # Case 2: Check if it's a directory with __init__.py
        elif os.path.isdir(possible_dir_path) and os.path.isfile(os.path.join(possible_dir_path, "__init__.py")):
            # Include all .py files in the module directory
            for root, _, files in os.walk(possible_dir_path):
                for file in files:
                    if file.endswith(".py"):
                        resolved_files.append(os.path.relpath(os.path.join(root, file), directory))

        # Check relative to the root directory
        possible_file_path = os.path.join(directory, module_path + ".py")
        possible_dir_path = os.path.join(directory, module_path)

        # Case 3: Check if it's a .py file
        if os.path.isfile(possible_file_path):
            resolved_files.append(os.path.relpath(possible_file_path, directory))

        # Case 4: Check if it's a directory with __init__.py
        elif os.path.isdir(possible_dir_path) and os.path.isfile(os.path.join(possible_dir_path, "__init__.py")):
            # Include all .py files in the module directory
            for root, _, files in os.walk(possible_dir_path):
                for file in files:
                    if file.endswith(".py"):
                        resolved_files.append(os.path.relpath(os.path.join(root, file), directory))
        else:
            print(f"[Not found]: '{module_name}' in {base_path}")
            
        return resolved_files

    # Step 2: Parse each Python file for imports and function calls
    for rel_path in file_list:
        if not match('|'.join(whitelist), rel_path):
            continue
        
        full_path = os.path.join(directory, rel_path)
        with open(full_path, "r", encoding="utf-8") as f:
            try:
                tree = ast.parse(f.read(), filename=rel_path)
            except SyntaxError as e:
                print(f"Syntax error in file {file}: {e}")
                continue
        
        def add_imports(file_path, module_name):
            module_files = resolve_module_to_files(file_path, module_name)
            for module_file in module_files:
                file_calls[file_path].add(module_file)
                file_called_by[module_file].add(file_path)
                
        # Analyze imports and function calls
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                # Handle `import module`
                for alias in node.names:
                    module_name = alias.name
                    add_imports(rel_path, module_name)

            elif isinstance(node, ast.ImportFrom):
                # Handle `from module import something`
                if node.module:
                    module_name = node.module
                    add_imports(rel_path, module_name)

    # Create a structured list of file relationships
    result = []
    for rel_path in file_list:
        # call graphs for white listed files
        result.append({
            "file": rel_path,
            "calls": list(file_calls.get(rel_path, [])),
            "called_by": list(file_called_by.get(rel_path, []))
        })

    return result


def main():
    # directory = "../data_repos/ftlr/datasets/frigate/code"
    directory = "../../NewsPolitics/newsscraper"

    if not os.path.isdir(directory):
        print("Invalid directory path.")
        return

    results = analyze_directory(directory)
    print('number of results', len(results))

    print("\nFile Relationships:")
    for file_info in results:
        print(f"File: {file_info['file']}")
        print(f"  Calls: {', '.join(file_info['calls']) or 'None'}")
        print(f"  Called by: {', '.join(file_info['called_by']) or 'None'}")


if __name__ == "__main__":
    main()