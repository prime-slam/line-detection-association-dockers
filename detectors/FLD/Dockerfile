FROM python:3.9-slim

ARG ALGORITHM_FOLDER=FLD

WORKDIR /detector

RUN apt-get update  \
    && apt-get install --no-install-recommends -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# install python requirements
COPY $ALGORITHM_FOLDER/requirements.txt requirements.txt

RUN python3 -m pip install --no-cache-dir -r requirements.txt

COPY $ALGORITHM_FOLDER .

COPY ./common ./common

ENTRYPOINT ["python3", "run.py"]