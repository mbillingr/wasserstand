FROM prefecthq/prefect:latest

RUN apt update

# required to autodetect file categories put into S3
RUN apt install media-types

WORKDIR /app

ADD . .

RUN pip install .
