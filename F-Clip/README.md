# F-Clip Docker Image
This folder contains adapters for [F-Clip](https://github.com/Delay-Xili/F-Clip) and a docker image
## Building Docker Image
1) Install the appropriate NVIDIA drivers with at least `cuda` version 11.3 on your host machine. You can check your cuda version with `nvidia-smi` command.
2) Install [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) on your host machine.
3) Build docker image using `Dockerfile`:
```
docker build -t fclip .
```
## Running Docker Container
To run the container use the following command:
```
sudo docker run --rm --gpus=all \
--mount type=bind,source=<IMAGES_PATH>,target=/app/input \
--mount type=bind,source=<OUTPUT_PATH>,target=/app/output \
fclip
```

Here `<IMAGES_PATH>` is the path where your image dataset is stored on the host machine and `<OUTPUT_PATH>` is the path where the results will be saved. 