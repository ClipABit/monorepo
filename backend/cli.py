import subprocess

def serve():
    subprocess.run(["modal", "serve", "main.py"])
