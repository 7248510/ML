@echo off
ECHO This script sets up a virtual environment for machine learning
python -m venv env
cd env/Scripts
ECHO cd ../../
activate
REM After activating the env the script does not pass the commands. This is a partial script!
REM cd ../../
REM UNCOMMENT LINES BELOW To set up the environment/install the prerequisites
REM py -m pip install --user virtualenv
REM py -m pip install --upgrade pip
REM py -m pip install wheel
REM py -m pip install -U setuptools
REM python -m pip install --upgrade requests
REM python -m pip install --upgrade -r requirements.txt
