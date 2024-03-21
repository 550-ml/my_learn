
import torch

from src.mask import mask_pad, mask_tril
from util.util import MultiHead, PositionEmbedding, FullyConnectOutput

# 编码器层
class EncoderLayer(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.mh = MultiHead()
        self.fc = FullyConnectOutput()
        
    def forward(self, x, mask):
        # x-> [b, 50, 32]
        score = self.mh(x, x, x, mask)
        
        # [b, 50, 32] -> [b, 50, 32]
        out = self.fc(score)
        
        return out
    
# 编码器
class Encoder(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layer_1 = EncoderLayer()
        self.layer_2 = EncoderLayer()
        self.layer_3 = EncoderLayer()
        
    def forward(self, x, mask):
        x = self.layer_1(x, mask)
        x = self.layer_2(x, mask)
        x = self.layer_3(x, mask)
        return x
    
# 解码器层
class DecoderLayer(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        
        self.mh1 = MultiHead()
        self.mh2 = MultiHead()
        
        self.fc = FullyConnectOutput()
        
    def forward(self, x, y, mask_pad_x, mask_tril_y):
        # calculate y's self attention
        # [b, 50, 32] -> [b, 50, 32]
        y = self.mh1(y, y, y, mask_tril_y)
        
        # calculate attention between x and y
        y = self.mh2(y, x, x, mask_pad_x)
        
        # [b, 50, 32] -> [b, 50, 32]
        y = self.fc(y)
        
        return y
    
# 解码器
class Decoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layer_1 = DecoderLayer()
        self.layer_2 = DecoderLayer()
        self.layer_3 = DecoderLayer()
        
    def forward(self, x, y, mask_pad_x, mask_tril_y):
        y = self.layer_1(x, y, mask_pad_x, mask_tril_y)
        y = self.layer_2(x, y, mask_pad_x, mask_tril_y)
        y = self.layer_3(x, y, mask_pad_x, mask_tril_y)
        return y
    
# Transformer
class Transformer(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embed_x = PositionEmbedding()
        self.embed_y = PositionEmbedding()
        self.encoder = Encoder()
        self.decoder = Decoder()
        self.fc_out = torch.nn.Linear(32, 39)
    
    def forward(self, x, y):
        mask_pad_x = mask_pad(x)
        mask_tril_y = mask_tril(y)
        
        # position embedding
        # x = [b, 50] -> [b, 50, 32]
        # y = [b, 50] -> [b, 50, 32]
        x, y = self.embed_x(x), self.embed_y(y)
        
        # encoder
        x = self.encoder(x, mask_pad_x)
        
        # decoder
        y = self.decoder(x, y, mask_pad_x, mask_tril_y)
        
        y = self.fc_out(y)
        return y