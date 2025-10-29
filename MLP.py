import torch
import torch.nn as nn
import torch.nn.functional as F



class MLP(nn.Module):
    """
    Linear Embedding
    """
    def __init__(self, input_dim, embed_dim, device=None):
        super().__init__()
        self.embed = embed_dim
        self.proj = nn.Linear(input_dim, embed_dim, device=device)

    def forward(self, x):
        b, n, w, h = x.shape
        x = x.flatten(2)       #[1, 512, 64]
        # print (f'x.shape={x.shape}')
        
        x = x.transpose(1, 2)  #[1, 64, 512]
        # print (f'x.shape={x.shape}')
        
        x = self.proj(x)
        # print (f'x.shape={x.shape}')  #[1, 64, 8]
        
        x = x.transpose(1, 2)
        # print (f'x.shape={x.shape}')  #[1, 8, 64]
        
        x = x.reshape(1, self.embed , w, h)
        # print (f'x.shape={x.shape}')  #[1, 8, 8, 8]

        # x = F.interpolate(x, 8)#, scale_factor, mode, align_corners)
        return x
    


if __name__ == '__main__':
    # device = 'cuda' if torch.cuda.is_available() else 'cpu'
    x = torch.randn(1, 512, 8, 8)

    model = MLP(input_dim=512, embed_dim=8)
    out = model(x)
    print(out.shape)
