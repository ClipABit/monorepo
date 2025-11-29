import os
import subprocess

def serve():
    subprocess.run(["modal", "serve", "main.py"])

def deploy():
    env = os.environ.copy()
    env["ENV"] = "prod"
    subprocess.run(["modal", "deploy", "main.py"], env=env)
