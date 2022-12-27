FROM python:3.10 AS base
COPY model_data model_data
ADD denoising.py denoising.py
ADD main.py server.py

# Setup env
ENV LANG C.UTF-8
ENV LC_ALL C.UTF-8
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONFAULTHANDLER 1


FROM base AS python-deps

# Install pipenv and compilation dependencies
RUN pip install pipenv

ADD Pipfile Pipfile
ADD Pipfile.lock Pipfile.lock

RUN apt-get update
RUN apt-get install ffmpeg libsm6 libxext6  -y

RUN PIPENV_VENV_IN_PROJECT=1 pipenv install --deploy
RUN pipenv install opencv-python




EXPOSE 80
ENTRYPOINT ["python3", "server.py"]