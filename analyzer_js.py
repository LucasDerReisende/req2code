import os
import subprocess
from collections import defaultdict
import json

from tqdm import tqdm


def parse_js_ts_file(file_path):
    """Parse a JavaScript/TypeScript file using @typescript-eslint/parser and return its AST."""
    # Execute a Node.js script to parse the file with @typescript-eslint/parser
    node_script = f"""
    try {{
    
    const fs = require('fs');
    const {{ parse }} = require('@typescript-eslint/parser');

    const code = fs.readFileSync('{file_path}', 'utf8');
    const ast = parse(code, {{ 
        sourceType: 'module', 
        ecmaFeatures: {{ jsx: true }},
        ecmaVersion: 'latest',
    }});
    console.log(JSON.stringify(ast));
    }} catch (error) {{
    }}
    """
    result = subprocess.run(
        ['node', '-e', node_script],
        check=True,
        capture_output=True,
        text=True
    )
    if not result.stdout.strip():
        return {}
    return json.loads(result.stdout)


def resolve_import_path(base_path, import_source, file_paths):
    """
    Resolve the full path of an import source.
    Args:
        base_path (str): The directory of the file where the import is found.
        import_source (str): The source string from the ImportDeclaration.
        file_mapping (dict): A mapping of filenames to their full paths.
    Returns:
        str: The resolved full path of the import or None if it cannot be resolved.
    """
    # Ignore npm packages and standard libraries (paths without ./ or ../)
    if not import_source.startswith('./') and not import_source.startswith('../'):
        return None

    # Resolve the relative path to an absolute path
    resolved_path = os.path.abspath(os.path.join(base_path, import_source))

    # Check if the resolved path points to an existing file in the project
    for extension in ['.js', '.ts']:
        full_path = resolved_path + extension
        if full_path in file_paths:
            return full_path

    return None

def analyze_directory(directory):
    """
    Analyze a directory of JS/TS files to find file import/export relationships.
    Args:
        directory (str): Path to the directory to analyze.
    Returns:
        list: A list of dictionaries, each representing a file with its imports and imported_by relationships.
    """
    file_imports = defaultdict(set)
    file_imported_by = defaultdict(set)
    # file_mapping = {}

    file_paths = []
    for root, _, files in os.walk(directory):
        for file in files:
            full_path = os.path.join(root, file)
            file_paths.append(full_path)

    # Step 2: Parse each JS/TS file for imports
    for path in tqdm(file_paths):
        if path.endswith(".js") or path.endswith(".ts") or path.endswith(".jsx") or path.endswith(".tsx"):
            ast_data = parse_js_ts_file(path)
            base_path = os.path.dirname(path)
            # Walk through the AST nodes to find import declarations
            for node in ast_data.get('body', []):
                if node['type'] == 'ImportDeclaration':
                    # Handle `import ... from 'module'`
                    import_source = node['source']['value']
                    resolved_path = resolve_import_path(base_path, import_source, file_paths)
                    if resolved_path:
                        file_imports[path].add(resolved_path)
                        file_imported_by[resolved_path].add(path)

    # Create a structured list of file relationships
    result = []
    for path in file_paths:
        result.append({
            "file": path,
            "calls": list(file_imports[path]),
            "called_by": list(file_imported_by[path])
        })

    return result

def main():
    directory = "/Users/lucas/Downloads/jitsi-meet-master"
    if not os.path.isdir(directory):
        print("Invalid directory path.")
        return

    results = analyze_directory(directory)
    print('Number of results:', len(results))
    print("\nFile Relationships:")
    for file_info in results:
        print(f"File: {file_info['file']}")
        print(f" Imports: {', '.join(file_info['calls']) or 'None'}")
        print(f" Imported by: {', '.join(file_info['called_by']) or 'None'}")

if __name__ == "__main__":
    main()

