# Python support can be specified down to the minor or micro version
# (e.g. 3.6 or 3.6.3).
# OS Support also exists for jessie & stretch (slim and full).
# See https://hub.docker.com/r/library/python/ for all supported Python
# tags from Docker Hub.

# If you prefer miniconda:
FROM continuumio/miniconda3:4.8.2

LABEL Name=rl-10 Version=0.0.1

WORKDIR /app
ADD ./requirements.txt /app

# Using pip:
RUN python3 -m pip install -r requirements.txt

ADD *.py /app/
CMD ["python3", "main.py"]

# CMD exec /bin/bash -c "trap : TERM INT; sleep infinity & wait"
