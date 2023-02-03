import torch

images = torch.rand(2, 1, 512, 512)
image1 = torch.rand(512, 512)
image2 = torch.rand(512, 512)
print(torch.stack([image1[None], image1[None]]).shape)
