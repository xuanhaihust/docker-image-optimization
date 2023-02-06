FROM bitnami/pytorch:1.13.1

# switch to root user
USER root

# basic utils (ncurses-bin for 'clear' command)
RUN apt update && apt install --no-install-recommends  -y \
    curl telnet ncurses-bin \
    locales ffmpeg libsm6 libxext6

# switch back to non-root user
USER 1001

RUN locale-gen en_US.UTF-8
ENV LANG en_US.UTF-8
ENV PYTHONIOENCODING UTF-8

COPY requirements.txt /tmp/
RUN pip install --no-cache-dir -r /tmp/requirements.txt

ADD app /code/app
WORKDIR /code