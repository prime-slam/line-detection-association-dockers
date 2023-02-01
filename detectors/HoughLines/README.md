# HoughLines Docker Image
This folder contains adapters for [HoughLines](https://docs.opencv.org/3.4/dd/d1a/group__imgproc__feature.html#ga46b4e588934f6c8dfd509cc6e0e4545a) and a docker image
## Building Docker Image
Build docker image using `Dockerfile`:
```
docker build -t houghlines .
```
Optionally, you can add your user to the docker group as described [here](https://docs.docker.com/engine/install/linux-postinstall/) so that running docker does not require root rights.
## Running Docker Container
To run the container use the following command:
```
docker run --rm \
-v <IMAGES_PATH>:/detector/input \
-v <OUTPUT_PATH>:/detector/output \
-v $(realpath ../common/):/detector/common \
houghlines [OPTIONAL_ARGS]
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
```