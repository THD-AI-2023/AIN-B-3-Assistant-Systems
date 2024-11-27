import os
import json
import fnmatch
import mimetypes

# Define the extensions that require partial content
PARTIAL_READ_EXTENSIONS = {'.csv', '.jsonl', '.txt', '.log'}  # Add more extensions as needed

# Initialize explicit filenames to include as "..." in JSON
EXPLICIT_IGNORE_FILES = {
    'LICENSE', 'CHANGELOG.md', 'README.md', '.dockerignore', '.gitignore',
    '.ignore', 'a_personas.md', 'b_use_cases.md', 'c_outlier_handling.md',
    'd_chatbot_use_case.md', 'e_sample_dialogs.md', 'f_dialog_flow.md',
    'endpoints.yml', 'dir_to_json.py', 'nlu.yml', 'config.yml', 'domain.yml',
    'rules.yml', 'stories.yml', 'credentials.yml'
}  # Default filenames

def parse_ignore_file(ignore_file_path):
    """
    Parses an ignore file and returns a set of filenames or patterns to ignore.
    Ignores empty lines and comments starting with '#'.
    """
    ignored_patterns = set()
    try:
        with open(ignore_file_path, 'r', encoding='utf-8') as file:
            for line in file:
                line = line.strip()
                if line and not line.startswith('#'):
                    # Normalize the pattern
                    normalized_pattern = os.path.normpath(line)
                    ignored_patterns.add(normalized_pattern)
    except FileNotFoundError:
        pass  # If ignore file does not exist, proceed with default ignore list
    return ignored_patterns

# Optionally, extend the ignore list by parsing an external .ignore file
IGNORE_FILE_PATH = os.path.join('.', '.ignore')  # Adjust the path if necessary
IGNORE_PATTERNS = parse_ignore_file(IGNORE_FILE_PATH)

def read_partial_file(file_path, first_n=10, last_m=5):
    """
    Reads the first `first_n` lines and the last `last_m` lines of a file.
    Inserts '...' if the file has more than `first_n + last_m` lines.
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            first_lines = []
            last_lines = []
            for line in file:
                if len(first_lines) < first_n:
                    first_lines.append(line.rstrip('\n').replace('`', '~'))
                last_lines.append(line.rstrip('\n').replace('`', '~'))
                if len(last_lines) > last_m:
                    last_lines.pop(0)

            total_lines = len(first_lines) + len(last_lines)
            # Estimate if there are more lines than first_n + last_m
            # This is a simplistic check; for large files, consider a different approach
            with open(file_path, 'r', encoding='utf-8') as f:
                total = sum(1 for _ in f)
            if total > first_n + last_m:
                return '\n'.join(first_lines) + '\n...\n' + '\n'.join(last_lines)
            else:
                return '\n'.join(first_lines + last_lines)
    except Exception:
        return "..."

def read_file(file_path):
    """
    Reads a file and returns its content appropriately, handling different file types.
    For specified extensions, it returns a partial content with '...'.
    For image files, it returns '...'.
    For other text files, it returns the full content with backticks replaced by tildes.
    """
    if os.path.basename(file_path) in EXPLICIT_IGNORE_FILES:
        return "..."
    try:
        mime_type, _ = mimetypes.guess_type(file_path)
        if mime_type and mime_type.startswith('image'):
            return "..."

        _, ext = os.path.splitext(file_path)
        if ext.lower() in PARTIAL_READ_EXTENSIONS:
            return read_partial_file(file_path)

        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()
            return content.replace('`', '~')
    except (UnicodeDecodeError, FileNotFoundError):
        return "..."

def is_ignored(file_path, ignore_patterns):
    """
    Checks if a file path matches any of the ignore patterns.
    Handles directory patterns by appending a '/' for accurate matching.
    """
    normalized_path = os.path.normpath(file_path)
    for pattern in ignore_patterns:
        normalized_pattern = os.path.normpath(pattern)
        if pattern.endswith('/'):
            # Directory pattern
            if fnmatch.fnmatch(normalized_path + '/', normalized_pattern):
                return True
        else:
            if fnmatch.fnmatch(normalized_path, normalized_pattern):
                return True
    return False

def is_in_submodule(file_path, submodule_paths):
    """
    Checks if a file path is within any of the submodule paths.
    """
    normalized_path = os.path.normpath(file_path)
    for submodule_path in submodule_paths:
        normalized_submodule = os.path.normpath(submodule_path)
        if normalized_path.startswith(normalized_submodule + os.sep):
            return True
    return False

def dir_to_json(directory, submodules, ignore_submodules=False):
    """
    Converts a directory structure into a JSON object, optionally ignoring submodules.
    """
    result = {}
    submodule_paths = [os.path.normpath(os.path.join(directory, submodule['path'])) for submodule in submodules]

    # Initialize ignore patterns
    global_ignore_patterns = set(IGNORE_PATTERNS)
    global_gitignore_patterns = set()

    for root, dirs, files in os.walk(directory):
        # Skip .git directories
        if '.git' in dirs:
            dirs.remove('.git')

        # Compute the relative path from the base directory
        relative_root = os.path.relpath(root, directory)
        if relative_root == ".":
            relative_root = ""

        # Handle submodule ignoring
        if ignore_submodules and is_in_submodule(root, submodule_paths):
            # Optionally, skip entire submodule directories
            dirs[:] = []  # Prevent walking into subdirectories
            continue

        # Parse .ignore and .gitignore files in the current directory
        ignore_file_path = os.path.join(root, '.ignore')
        gitignore_file_path = os.path.join(root, '.gitignore')

        local_ignore_patterns = parse_ignore_file(ignore_file_path)
        local_gitignore_patterns = parse_ignore_file(gitignore_file_path)

        # Update global ignore patterns
        global_ignore_patterns.update(local_ignore_patterns)
        global_gitignore_patterns.update(local_gitignore_patterns)

        # Remove ignored directories from traversal
        dirs_to_remove = []
        for d in dirs:
            dir_relative_path = os.path.normpath(os.path.join(relative_root, d))
            if is_ignored(dir_relative_path, global_ignore_patterns):
                dirs_to_remove.append(d)
        for d in dirs_to_remove:
            dirs.remove(d)

        # Navigate to the correct location in the result dictionary
        sub_result = result
        if relative_root:
            for part in relative_root.split(os.sep):
                sub_result = sub_result.setdefault(part, {})

        for file in files:
            file_path = os.path.join(root, file)
            relative_file_path = os.path.relpath(file_path, directory)
            relative_file_path = os.path.normpath(relative_file_path)

            # Check if file should be completely ignored
            if is_ignored(relative_file_path, global_ignore_patterns):
                continue  # Do not include in JSON at all

            # Check if file should be included as "..." (from .gitignore or explicit ignore)
            if is_ignored(relative_file_path, global_gitignore_patterns) or file in EXPLICIT_IGNORE_FILES:
                sub_result[file] = "..."
                continue

            # Handle submodules
            if ignore_submodules and is_in_submodule(file_path, submodule_paths) and 'README' not in file:
                continue

            # Read and assign file content
            file_content = read_file(file_path)
            sub_result[file] = file_content

    return result

def load_submodules(gitmodules_path):
    """
    Loads submodules from the .gitmodules file.
    """
    submodules = []
    try:
        with open(gitmodules_path, 'r', encoding='utf-8') as gitmodules_file:
            current_submodule = {}
            for line in gitmodules_file:
                line = line.strip()
                if line.startswith('[submodule'):
                    if current_submodule:
                        submodules.append(current_submodule)
                        current_submodule = {}
                elif line.startswith('path') and '=' in line:
                    _, path = line.split('=', 1)
                    current_submodule['path'] = path.strip()
            if current_submodule:
                submodules.append(current_submodule)
    except FileNotFoundError:
        pass  # No submodules present
    return submodules

def main(directory='.', ignore_submodules=False, output_file='output.json'):
    """
    Main function to generate the directory structure in JSON format.
    """
    gitmodules_path = os.path.join(directory, '.gitmodules')
    submodules = load_submodules(gitmodules_path)

    json_data = dir_to_json(directory, submodules, ignore_submodules)
    json_output = os.path.join(directory, output_file)

    try:
        with open(json_output, 'w', encoding='utf-8') as json_file:
            json.dump(json_data, json_file, ensure_ascii=False, indent=4)
        print(f"JSON data has been written to {json_output}")
    except Exception as e:
        print(f"Failed to write JSON data to {json_output}: {e}")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Convert directory structure to JSON.")
    parser.add_argument(
        '-d', '--directory',
        type=str,
        default='.',
        help='Directory to convert (default: current directory)'
    )
    parser.add_argument(
        '-i', '--ignore-submodules',
        action='store_true',
        help='Ignore submodule directories'
    )
    parser.add_argument(
        '-o', '--output',
        type=str,
        default='output.json',
        help='Output JSON file name (default: output.json)'
    )

    args = parser.parse_args()
    main(directory=args.directory, ignore_submodules=args.ignore_submodules, output_file=args.output)
