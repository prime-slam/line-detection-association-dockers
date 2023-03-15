# FLD Docker Image
This folder contains adapters for [FLD](https://docs.opencv.org/4.x/df/ded/group__ximgproc__fast__line__detector.html) and a docker image
## Building Docker Image
Build docker image using `Dockerfile`:
```
docker build -t fld -f Dockerfile ..
```
Optionally, you can add your user to the docker group as described [here](https://docs.docker.com/engine/install/linux-postinstall/) so that running docker does not require root rights.
## Running Docker Container
To run the container use the following command:
```
docker run --rm \
-v <IMAGES_PATH>:/detector/input \
-v <OUTPUT_PATH>:/detector/output \
fld [OPTIONAL_ARGS]
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