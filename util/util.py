import math

import torch

# 注意力计算函�?
def attention(Q, K, V, mask):
    # b句话，每句话50个词，每个词32维向量，四个头就�?8维向�?
    # 这里我自己理解一下，我们是需要多个W_q, W_k, W_v的，但是我们只有一个输入后的embedding�?32维的向量可以表示一个单词，同理8个长度的单词也可以表示一个单词，所以我们需要将32维的向量分成4�?8维的向量，然后再进行矩阵乘法，这样就相当于有�?4个W_q, W_k, W_v
    
    # [b, 4, 50, 8] * [b, 4, 8 ,50] -> [b, 4, 50, 50]
    # 一个句子中，每个单词对其他单词的注意力分数
    score  = torch.matmul(Q, K.transpose(-1, -2))

    # 数值缩�?
    score = score / math.sqrt(8)
    
    # mask,mask为true的地方是负无穷，经过softmax后为0
    # mask = [b, 1, 50, 50]
    score = score.masked_fill_(mask, -float('inf'))
    score = score.softmax(dim=-1)
    
    # 以注意力分数乘以V，得到最终的注意力结�?
    # [b, 4, 50, 50] * [b, 4, 50, 8] -> [b, 4, 50, 8]
    score = torch.matmul(score, V)
    
    # 多个注意力头相加
    # [b, 4, 50, 8] -> [b, 50, 32]
    score = score.transpose(1,2).reshape(-1,50,32)
    
    return score

# 多头注意力层，输入的�?32，里面拆分做多头
class MultiHead(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc_Q = torch.nn.Linear(32, 32)
        self.fc_K = torch.nn.Linear(32, 32)
        self.fc_V = torch.nn.Linear(32, 32)
        
        self.out_fc = torch.nn.Linear(32, 32)
        
        self.norm =     torch.nn.LayerNorm(normalized_shape=32,
                                           elementwise_affine=True)
        self.dropout = torch.nn.Dropout(p=0.1)
    def forward(self, Q, K, V, mask):
        # b句话，每句话50个单词，每个单词�?32维向�?
        # Q,K,V = [b, 50, 32]
        b = Q.shape[0]
        
        clone_Q = Q.clone()
        
        # 规范化输�?
        Q = self.norm(Q)
        K = self.norm(K)
        V = self.norm(V)
        
        # 线性运�?
        Q = self.fc_Q(Q)
        K = self.fc_K(K)
        V = self.fc_V(V)
        
        # 拆分多头
        # [b, 50, 32] -> [b, 50, 4, 8] -> [b, 4, 50, 8]
        Q = Q.reshape(b, 50, 4, 8).transpose(1,2)
        K = K.reshape(b, 50, 4, 8).transpose(1,2)
        V = V.reshape(b, 50, 4, 8).transpose(1,2)
        
        # 计算注意�?
        # [b, 4, 50, 8] -> [b, 50, 32]
        score = attention(Q, K, V, mask)
        
        score = self.dropout(self.out_fc(score))

        # 短接
        score = score + clone_Q
        
        return score

# 位置编码�?
class PositionEmbedding(torch.nn.Module):
    def __init__(self):
        super().__init__()
        
        def get_pe(pos, i, d_model):
            fenmu = 1e4 ** (i / d_model)
            pe = pos / fenmu
            
            if i % 2 == 0:
                return math.sin(pe)
            return math.cos(pe)
        
        # 初始化位置矩�?
        pe = torch.empty(50, 32)
        for i in range(50):
            for j in range(32):
                pe[i, j] = get_pe(i, j, 32)
        pe = pe.unsqueeze(0)
        
        # 定义为不更新的常�?
        self.register_buffer('pe', pe)
        
        # 词编�?
        self.embed = torch.nn.Embedding(39, 32)
        # 初始化参�?
        self.embed.weight.data.normal_(mean=0.0, std=0.02)
        
    def forward(self, x):
        # [8, 50] -> [8, 50, 32]
        embed = self.embed(x)
        
        # 词编码和位置编码相加
        # [8, 50, 32] + [1, 50, 32] -> [8, 50, 32]
        embed = embed + self.pe
        return embed
    
# 全连接输出层
class FullyConnectOutput(torch.nn.Module):
    def __init__(self):
        super().__init__()
        
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(32, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 32),
            torch.nn.Dropout(p=0.1),  
        )
        self.norm = torch.nn.LayerNorm(32, elementwise_affine=True)
    def forward(self, x):
        # 短接
        clone_x = x.clone()
        
        # 规范�?
        x = self.norm(x)

        # [8, 50, 32] -> [8, 50, 32]
        x = self.fc(x)

        # 短接
        out = clone_x + x
        return out
tem=1
print(tem)

