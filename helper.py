import os

def getGitPath():
    current_dir = os.path.dirname(os.path.realpath(__file__))
    git_dir = os.path.dirname(os.path.dirname(current_dir))
    return git_dir