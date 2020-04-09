# Python support can be specified down to the minor or micro version
# (e.g. 3.6 or 3.6.3).
# OS Support also exists for jessie & stretch (slim and full).
# See https://hub.docker.com/r/library/python/ for all supported Python
# tags from Docker Hub.
FROM python:3.7.6

# If you prefer miniconda:
#FROM continuumio/miniconda3

LABEL Name=rl-06 Version=0.0.1
EXPOSE 3000

WORKDIR /app
ADD ./requirements.txt /app

# Using pip:
RUN python3 -m pip install -r requirements.txt

ADD *.py /app/
CMD ["python3", "main.py"]

# CMD exec /bin/bash -c "trap : TERM INT; sleep infinity & wait"



### Exploring options for running scripts as layers
## Logic: If an earlier script is updated, 
#    all subsequent scripts need to be updated
#    Can't output to a volume
#    Could have CMD simply copy generated outputs to disk!

# RUN python3 10_comparing_epsilons.py
# RUN python3 11_optimistic_initial_values.py
# RUN python3 12_ucb1.py
