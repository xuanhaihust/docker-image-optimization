FROM python:3.8.15

# opencv
RUN apt update && apt install -y locales ffmpeg libsm6 libxext6
RUN apt update && apt install -y libsm6 libxext6 libxrender-dev libglib2.0-0 poppler-utils build-essential libpoppler-cpp-dev pkg-config

# healthcheck and compression
RUN apt update && apt install -y curl && apt install -y unrar-free

RUN locale-gen en_US.UTF-8
ENV LANG en_US.UTF-8
ENV PYTHONIOENCODING UTF-8

COPY requirements.txt /tmp/
RUN pip install -r /tmp/requirements.txt

# nvidia serving client
ADD wheel-installs /tmp/wheels
RUN pip install /tmp/wheels/*
RUN pip install tritonclient[all]==2.19.0

ADD app /code/app
WORKDIR /code