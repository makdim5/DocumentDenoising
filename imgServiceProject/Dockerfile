FROM python:3.10 AS base
COPY model_data model_data
ADD denoising.py denoising.py
ADD main.py server.py
ADD requirements.txt requirements.txt

# Setup env
ENV LANG C.UTF-8
ENV LC_ALL C.UTF-8
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONFAULTHANDLER 1


FROM base AS python-deps


RUN apt-get update
RUN apt-get install ffmpeg libsm6 libxext6  -y

RUN pip install -r requirements.txt


EXPOSE 80
ENTRYPOINT ["python3", "server.py"]