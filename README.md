# line-detection-dockers

This repository provides adapted and docker-wrapped line segments detection algorithms.
### Structure
Each folder contains an adapted algorithm, a `Dockerfile`, and instructions for building and running.

### Adapted algorithms
The following detectors are currently adapted.


| Name    | Paper | Original implementation                         |
|---------| --- | --- |
| F-Clip  | [Fully Convolutional Line Parsing](https://arxiv.org/abs/2104.11207v2) | [Code](https://github.com/Delay-Xili/F-Clip) |
| HAWP    | [Holistically-Attracted Wireframe Parsing: From Supervised to Self-Supervised Learning](https://arxiv.org/abs/2210.12971) | [Code](https://github.com/cherubicXN/hawp) |
| L-CNN   | [End-to-End Wireframe Parsing](https://arxiv.org/abs/1905.03246) | [Code](https://github.com/zhou13/lcnn) |
| HT-LCNN | [Deep Hough-Transform Line Priors](https://arxiv.org/abs/2007.09493) | [Code](https://github.com/yanconglin/Deep-Hough-Transform-Line-Priors) |
