import torch

from src.data import zidian_y, loader, zidian_xr, zidian_yr
from src.mask import mask_pad, mask_tril # mask_pad:padding mask, mask_tril:triangle mask
from src.model import Transformer

model = Transformer()
loss_func = torch.nn.CrossEntropyLoss()
optim = torch.optim.Adam(model.parameters(), lr=2e-3)
# learning rate  decay
sched = torch.optim.lr_scheduler.StepLR(optim, step_size=3, gamma=0.5)
NUM_EPOCHS = 1

for epoch in range(NUM_EPOCHS):
    for index, (x,y) in enumerate(loader):
        # x = [b, 50]
        # y = [b, 51] 
        
        # we predict the next word based on the previous words, but the last word dont have next word, so we dont need the last word
        # [b, 50, 39] '39'is the 概率 of each word
        pred = model(x, y[:, :-1])
        
        # [b, 50, 39] -> [b*50, 39]
        pred = pred.reshape(-1, 39)
        
        # [b, 51] -> [b*50]
        y = y[:, 1:].reshape(-1)
        test =1
        # ignore pad
        select = y != zidian_y['<PAD>']
        pred = pred[select]
        y = y[select]
        
        loss = loss_func(pred, y)
        optim.zero_grad()
        loss.backward()
        optim.step()
        
        # print loss
        if index % 200 == 0:
            pred = pred.argmax(dim=1)
            correct = (pred == y).sum().item()
            accuracy = correct / len(pred)
            lr = optim.param_groups[0]['lr']
            print(epoch, index, lr, loss.item(), accuracy)
    
    sched.step()
    
# predict
def predict(x):
    model.eval()
    
    # [1, 50]
    mask_pad_x = mask_pad(x)
    
    # init y,begin with <SOS>
    # [1, 50]
    target = [zidian_y['<SOS>']] + [zidian_y['<PAD>']] * 49
    target = torch.LongTensor(target).unsqueeze(0)
    # [[0,2,2,2]]

    #  embedding
    # [1, 50] -> [1, 50, 32]
    x = model.embed_x(x)
    
    # encoding
    x = model.encoder(x, mask_pad_x)
    
    # generate y is similar to beam search
    for i in range(49):
        # [1, 50]
        y = target
        
        # tril mask
        mask_tril_y = mask_tril(y)
        
        # embedding
        y = model.embed_y(y)
        
        y = model.decoder(x, y, mask_pad_x, mask_tril_y)
        
        # [1, 50, 32] -> [1, 50, 39]
        out = model.fc_out(y)
        
        out = out[:, i, :]
        
        out = out.argmax(dim=1).detach()
        
        target[:, i+1] = out
        
    return target

# test
for i, (x,y) in enumerate(loader):
    break

for i in range(8):
    print(i)
    print(''.join([zidian_xr[i] for i in x[i].tolist()]))
    print(''.join([zidian_yr[i] for i in y[i].tolist()]))
    print(''.join([zidian_yr[i] for i in predict(x[i].unsqueeze(0))[0].tolist()]))


    