# FROM rocker/tidyverse:3.6.2
FROM rocker/r-ver:3.6.2

LABEL Name=rl-04 Version=0.0.1

WORKDIR /app

# Set up requirements
ADD ./requirements.R /app
RUN Rscript requirements.R

ADD *.R /app/
ADD ./R/* /app/R/
CMD ["Rscript", "main.R"]
#CMD ["R""]
