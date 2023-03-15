FROM python:3.9-slim

WORKDIR /detector

RUN apt-get update  \
    && apt-get install --no-install-recommends -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# install python requirements
COPY ./HoughLinesP/requirements.txt requirements.txt

RUN python3 -m pip install --no-cache-dir -r requirements.txt

COPY ./HoughLinesP .

COPY ./common ./common

ENTRYPOINT ["python3", "run.py"]