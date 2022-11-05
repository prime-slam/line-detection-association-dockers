FROM pytorch/pytorch:1.11.0-cuda11.3-cudnn8-runtime
WORKDIR /app

# OpenCV dependencies
RUN apt-get update && apt-get install --no-install-recommends -y \
    libgl1 \
    libglib2.0-0

# download pretrained models
RUN python3 -m pip install --no-cache-dir gdown  \
    && gdown "1jMZB-kYTNAGaVPW0FBLTAQMc3m3vR-_P" \
    && apt-get install -y --no-install-recommends unzip \
    && unzip ckpt.zip \
    && rm ckpt.zip \
    && rm -rf __MACOSX \
    && mv ckpt pretrained

COPY requirements.txt requirements.txt
RUN python3 -m pip install --no-cache-dir -r requirements.txt

COPY . .

ENTRYPOINT ["python3", "run.py"]