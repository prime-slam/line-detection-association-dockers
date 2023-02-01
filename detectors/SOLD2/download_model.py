import torch

if __name__ == "__main__":
    url = "https://www.polybox.ethz.ch/index.php/s/blOrW89gqSLoHOk/download"
    torch.hub.load_state_dict_from_url(url, map_location=lambda storage, loc: storage)
