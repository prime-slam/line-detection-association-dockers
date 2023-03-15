# LBD Docker Image
This folder contains adapter for [LBD](https://github.com/iago-suarez/pytlbd) and a docker image
## Building Docker Image
Build docker image using `Dockerfile`:
```
docker build -t lbd -f Dockerfile ..
```
Optionally, you can add your user to the docker group as described [here](https://docs.docker.com/engine/install/linux-postinstall/) so that running docker does not require root rights.
## Running Docker Container
To run the container use the following command:
```
docker run --rm \
-v <IMAGES_PATH>:/associator/input \
-v <LINES_PATH>:/associator/lines \
-v <OUTPUT_PATH>:/associator/output \
lbd [OPTIONAL_ARGS]
```

Here `<IMAGES_PATH>` is the path where your image dataset is stored on the host machine. `<LINES_PATH>` is the path where your predicted lines are stored on the host machine as CSV files (each line must be stored in the following format: `x1,y1,x2,y2`).`<OUTPUT_PATH>` is the path where the results will be saved.

The following `[OPTIONAL_ARGS]` can be used:
```
optional arguments:
  -h, --help            show this help message and exit
  --imgs PATH, -i PATH  path to images (default: input/)
  --lines PATH, -l PATH
                        path to predicted lines (default: lines/)
  --output PATH, -o PATH
                        output path (default: output/)
  --associations-dir STRING, -a STRING
                        name of associations output directory (default:
                        associations)
  --pairs PATH, -p PATH
                        path to a file with pairs of frames for which
                        associations will be calculated (default: None)
  --step NUM            step with which associations between frames will be
                        calculated (does not work if a file with pairs is
                        provided) (default: 1)
```