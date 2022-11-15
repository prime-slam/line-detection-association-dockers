# line-detection-dockers

This repository provides adapted and docker-wrapped line segments detection algorithms.
### Structure
Each folder contains an adapted algorithm, a `Dockerfile`, and instructions for building and running.

### Adapted algorithms
The following detectors are currently adapted.


| Name    | Paper | Original implementation                        |
|---------| --- | --- |
| F-Clip  | [Fully Convolutional Line Parsing](https://arxiv.org/abs/2104.11207v2) | [Code](https://github.com/Delay-Xili/F-Clip) |
| HAWP    | [Holistically-Attracted Wireframe Parsing: From Supervised to Self-Supervised Learning](https://arxiv.org/abs/2210.12971) | [Code](https://github.com/cherubicXN/hawp) |
| L-CNN   | [End-to-End Wireframe Parsing](https://arxiv.org/abs/1905.03246) | [Code](https://github.com/zhou13/lcnn) |
| HT-LCNN | [Deep Hough-Transform Line Priors](https://arxiv.org/abs/2007.09493) | [Code](https://github.com/yanconglin/Deep-Hough-Transform-Line-Priors) |
| TP-LSD  | [TP-LSD: Tri-Points Based Line Segment Detector](https://arxiv.org/abs/2009.05505) | [Code](https://github.com/Siyuada7/TP-LSD) |
| ULSD    | [ULSD: Unified Line Segment Detection across Pinhole, Fisheye, and Spherical Cameras](https://arxiv.org/abs/2011.03174) | [Code](https://github.com/lh9171338/Unified-Line-Segment-Detection) |
| LETR    | [Line Segment Detection Using Transformers without Edges](https://arxiv.org/abs/2101.01909) | [Code](https://github.com/mlpc-ucsd/LETR) |