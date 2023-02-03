import torch

if __name__ == "__main__":
    url = "http://cmp.felk.cvut.cz/~mishkdmy/models/sold2_wireframe.pth"
    torch.hub.load_state_dict_from_url(url, map_location=lambda storage, loc: storage)
