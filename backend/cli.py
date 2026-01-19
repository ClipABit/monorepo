import subprocess

def serve():
    subprocess.run(["modal", "serve", "apps/server.py"])
