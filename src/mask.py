import torch

from src.data import zidian_x, zidian_y

print(zidian_x)
def mask_pad(data):
    # b,each sentence 50 words
    # data = [b, 50]
    mask = data == zidian_x['<PAD>']
    
    # [b, 50] -> [b, 1, 1, 50]
    mask = mask.reshape(-1, 1, 1, 50)

    # [b, 1, 1, 50] -> [b, 1, 50, 50]
    mask = mask.expand(-1, 1, 50, 50)
    return mask
print(zidian_y)
def mask_tril(data):
    # data = [b, 50]
    # [1, 50, 50]
    # generate a 50*50 matrix to mask
    
    tril = 1 - torch.tril(torch.ones(1, 50, 50, dtype=torch.long))
    
    # [b, 50]
    mask = data == zidian_y['<PAD>']
    
    # [b, 1, 50]
    mask = mask.unsqueeze(1).long()
    
    # [b, 1, 50] + [1, 50, 50] -> [b, 50, 50]
    mask = mask+tril

    mask = mask > 0
    
    mask = (mask == 1).unsqueeze(dim=1)
    
    return mask
    