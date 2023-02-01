# HAWP Docker Image
This folder contains adapters for [HAWP](https://github.com/cherubicXN/hawp) and a docker image
## Building Docker Image
1) Install the appropriate NVIDIA drivers with at least `cuda` version 11.3 on your host machine. You can check your cuda version with `nvidia-smi` command.
2) Install [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) on your host machine.
3) Build docker image using `Dockerfile`:
```
docker build -t hawp .
```
Optionally, you can add your user to the docker group as described [here](https://docs.docker.com/engine/install/linux-postinstall/) so that running docker does not require root rights.
## Running Docker Container
To run the container use the following command:
```
docker run --rm --gpus=all \
-v <IMAGES_PATH>:/detector/input \
-v <OUTPUT_PATH>:/detector/output \
-v $(realpath ../common/):/detector/common \
hawp [OPTIONAL_ARGS]
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
  --model-config PATH, -m PATH
                        pretrained model configuration path (default: configs/hawpv2.yaml)
  --model PATH, -M PATH
                        pretrained model path (default: checkpoints/hawpv2-edb9b23f.pth)
  --device STRING, -d STRING
                        name of desired execution device: gpu or cpu (default: gpu)

```