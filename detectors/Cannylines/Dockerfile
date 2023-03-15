FROM python:3.9-slim

ARG CANNYLINES_PATH=https://github.com/ludlows/CannyLine/archive/refs/heads/master.zip

WORKDIR /detector

RUN apt-get update  \
    && apt-get install --no-install-recommends -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# install python requirements
COPY ./Cannylines/requirements.txt requirements.txt

RUN python3 -m pip install --no-cache-dir $CANNYLINES_PATH -r requirements.txt

COPY ./Cannylines .

COPY ./common ./common

ENTRYPOINT ["python3", "run.py"]