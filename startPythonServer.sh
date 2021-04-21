#!/bin/bash

cd /home/delta/Downloads/PythonServer/
source ./env/bin/activate
export FLASK_APP=run.py
export FLASK_ENV=production
/home/delta/anaconda3/bin/python -m flask run
