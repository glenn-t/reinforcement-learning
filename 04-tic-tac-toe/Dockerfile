# FROM rocker/tidyverse:3.6.2
FROM rocker/r-ver:3.6.2

LABEL Name=rl-04 Version=0.0.1

WORKDIR /app

# Set up requirements
ADD ./requirements.R /app
RUN Rscript requirements.R

ADD *.R /app/
CMD ["Rscript", "main.R"]

# CMD exec /bin/bash -c "trap : TERM INT; sleep infinity & wait"



### Exploring options for running scripts as layers
## Logic: If an earlier script is updated, 
#    all subsequent scripts need to be updated
#    Can't output to a volume
#    Could have CMD simply copy generated outputs to disk!

# RUN python3 10_comparing_epsilons.py
# RUN python3 11_optimistic_initial_values.py
# RUN python3 12_ucb1.py