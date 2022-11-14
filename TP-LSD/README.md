# TP-LSD Docker Image
This folder contains adapters for [TP-LSD](https://github.com/Siyuada7/TP-LSD/) and a docker image
## Building Docker Image
1) Install the appropriate NVIDIA drivers with at least `cuda` version 11.3 on your host machine. You can check your cuda version with `nvidia-smi` command.
2) Install [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) on your host machine.
3) Change default docker runtime to nvidia. See `Default runtime` [here](https://github.com/NVIDIA/nvidia-docker/wiki/Advanced-topics#default-runtime).
4) Build docker image using `Dockerfile`:
```
docker build -t tp-lsd .
```
Optionally, you can add your user to the docker group as described [here](https://docs.docker.com/engine/install/linux-postinstall/) so that running docker does not require root rights.
## Running Docker Container
To run the container use the following command:
```
docker run --rm --gpus=all \
-v <IMAGES_PATH>:/detector/input \
-v <OUTPUT_PATH>:/detector/output \
-v $(realpath ../common/):/detector/common \
tp-lsd [OPTIONAL_ARGS]
```

Here `<IMAGES_PATH>` is the path where your image dataset is stored on the host machine and `<OUTPUT_PATH>` is the path where the results will be saved. 

The following `[OPTIONAL_ARGS]` can be used:
```
optional arguments:
  -h, --help            show this help message and exit
  --imgs PATH, -i PATH  path to images (default: input/)
  --output PATH, -o PATH
                        output path (default: output/)
  --lines-dir STRING, -l STRING
                        name of lines output directory (default: lines)
  --scores-dir STRING, -s STRING
                        name of scores output directory (default: scores)
  --batch NUM, -b NUM   dataloader batch size (default: 1)
  --model-path PATH, -m PATH
                        pretrained model path (default: pretraineds/HR/checkpoint.pth.tar)
  --model STR, -M STR   Name of model: TPLSD, TPLSD_Lite, TPLSD_512 or Hourglass (default: TPLSD)
  --device STRING, -d STRING
                        name of desired execution device: cuda or cpu (default: cuda)

```