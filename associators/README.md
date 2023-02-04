# line-association-dockers

This folder contains adapted and docker-wrapped line segments association algorithms.
### Structure
Each folder contains an adapted algorithm, a `Dockerfile`, and instructions for building and running.

### Adapted algorithms
The following associators are currently adapted.


| Name        | Paper                                                                                                                                          | Original implementation                                    |
|-------------|------------------------------------------------------------------------------------------------------------------------------------------------|------------------------------------------------------------|
| LineTR      | [Line as a Visual Sentence: Context-aware Line Descriptor for Visual Localization](https://arxiv.org/abs/2109.04753)                           | [Code](https://github.com/yosungho/LineTR)         |
| DLD         | [DLD: A Deep Learning Based Line Descriptor for Line Feature Matching](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=8968062&casa_token=EU8AeQtFSSEAAAAA:Z08gSx-VGZs1t4SiYydyFdb6asrl4b5Hbx4wh-Y6lMxQ7RgMVoSRPr8x2Hern1pOCPTnlwD8Ap94&tag=1)    | [Code](https://github.com/manuellange/DLD)         |
| WLD         | [WLD: A Wavelet and Learning based Line Descriptor for Line Feature Matching](https://diglib.eg.org/handle/10.2312/vmv20201186)    | [Code](https://github.com/manuellange/WLD)         |
| SOLD2       | [SOLD2: Self-supervised Occlusion-aware Line Description and Detection](https://arxiv.org/abs/2104.03362)    | [Code](https://github.com/cvg/SOLD2)         |
| LBD         | [An efficient and robust line segment matching approach based on LBD descriptor and pairwise geometric consistency](https://www.sciencedirect.com/science/article/pii/S1047320313000874?casa_token=bnkB0vpZNLgAAAAA:3QR5bV0jqIIS82HMtasTxzbAhwO5TPIAbiGgcvLsajK7WTkmqnJkN_-5mqiJyhsxXzFaB6bAcQ)    | [Code](https://github.com/iago-suarez/pytlbd)         |
