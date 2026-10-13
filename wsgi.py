import os
import sys

# Ensure current project directory is in the Python search path and set as working directory
project_home = os.path.dirname(os.path.abspath(__file__))
if project_home not in sys.path:
    sys.path.insert(0, project_home)
os.chdir(project_home)

# Native WSGI Flask Application (instant 20ms response on PythonAnywhere)
from flask_app import app as application
