import os
import sys

# Ensure current project directory is in the Python search path and set as working directory
project_home = os.path.dirname(os.path.abspath(__file__))
if project_home not in sys.path:
    sys.path.insert(0, project_home)
os.chdir(project_home)

from a2wsgi import ASGIMiddleware
from main import app

# PythonAnywhere expects a WSGI callable named 'application'
application = ASGIMiddleware(app)
