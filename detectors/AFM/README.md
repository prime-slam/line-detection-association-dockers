# AFM Docker Image
This folder contains adapters for [AFM](https://github.com/cherubicXN/afm_cvpr2019) and a docker image
## Building Docker Image
1) Install the appropriate NVIDIA drivers with at least `cuda` version 11.0 on your host machine. You can check your cuda version with `nvidia-smi` command.
2) Install [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) on your host machine.
3) Build docker image using `Dockerfile`:
```
docker build -t afm -f Dockerfile ..
```
Optionally, you can add your user to the docker group as described [here](https://docs.docker.com/engine/install/linux-postinstall/) so that running docker does not require root rights.
## Running Docker Container
To run the container use the following command:
```
docker run --rm --gpus=all \
-v <IMAGES_PATH>:/detector/input \
-v <OUTPUT_PATH>:/detector/output \
afm [OPTIONAL_ARGS]
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
                        pretrained model configuration path (default: 
                        /detector/experiments/afm_atrous.yaml)
  --model PATH, -M PATH
                        pretrained model path (default: /detector/experiments/
                        atrous/weight/model_final.pth.tar)
  --device STRING, -d STRING
                        name of desired execution device: gpu or cpu (default: gpu)

```