FROM ubuntu:20.04

WORKDIR /projects

COPY Backend/ /projects

RUN apt update && \
    apt install -y sudo && \
    apt install python3 -y && \
    apt install python3-pip -y 

RUN pip install --no-cache-dir -r requirements.txt

ENV FLASK_APP=app
ENV FLASK_RUN_HOST=0.0.0.0

EXPOSE 5000