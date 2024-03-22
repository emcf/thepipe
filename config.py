import os
CTAGS_COMPATIBLE_EXTENSIONS = {'.h', '.c', '.cpp', '.java', '.py', '.cs'} # Ctags-compatible code files
LLMLINGUA_COMPATIBLE_EXTENSIONS = {'.txt', '.md', '.pdf', '.docx' } # Human-readable text-like files
TXT_EXTENSIONS = {'.txt', '.md', '.tex', '.json', '.yaml', '.xaml', '.ini', '.sh', '.xml', '.js', '.jsx', '.ts', '.tsx'} # Files to be read as plaintext
FILES_TO_IGNORE = {'.gitignore', '.bin', '.pyc', '.pyo', '.exe', '.dll', '.obj', '.o', '.a', '.lib', '.so', '.dylib', '.ncb', '.sdf', '.suo', '.pdb', '.idb', '.pyd', '.ipynb_checkpoints', '.npy', '.pth'} # Files to ignore, please feel free to customize!
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")