import torch

if __name__ == "__main__":
    url = "https://download.pytorch.org/models/resnet101-5d3b4d8f.pth"
    torch.hub.load_state_dict_from_url(url, map_location=lambda storage, loc: storage)
