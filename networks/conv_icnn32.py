import torch
import torch.nn as nn
from torch import autograd

#What does this layer do, I am not sure
class ConvexQuadratic(nn.Module):
    '''Convex Quadratic Layer'''
    __constants__ = ['in_features', 'out_features', 'quadratic_decomposed', 'weight', 'bias']

    def __init__(self, in_features, out_features, bias=True, rank=1):
        super(ConvexQuadratic, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        
        self.quadratic_decomposed = nn.Parameter(torch.Tensor(
            torch.randn(in_features, rank, out_features)
        ))
        self.weight = nn.Parameter(torch.Tensor(
            torch.randn(out_features, in_features)
        ))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)

    def forward(self, input):
        quad = ((input.matmul(self.quadratic_decomposed.transpose(1,0)).transpose(1, 0)) ** 2).sum(dim=1)
        linear = F.linear(input, self.weight, self.bias)
        return quad + linear
    
class View(nn.Module):
    def __init__(self, *shape):
        super(View, self).__init__()
        self.shape = shape
    def forward(self, input):
        #print(input.shape)
        return input.view(-1, *self.shape)

class ConvICNN128(nn.Module):
    def __init__(self, channels=3):
        super(ConvICNN128, self).__init__()

        self.first_linear = nn.Sequential(
            nn.Conv2d(channels, 128, kernel_size=3, padding=1, bias=False),
            nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False),
            nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False),
            nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=True),
        )
        
        self.first_squared = nn.Sequential(
            nn.Conv2d(channels, 128, kernel_size=3, padding=1, bias=False),
            nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False),
            nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False),
            nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False),
        )
        
        self.convex = nn.Sequential(
            nn.CELU(),
            nn.Conv2d(128, 128, kernel_size=3,stride=2, bias=True, padding=1),  
            nn.CELU(),
            nn.Conv2d(128, 128, kernel_size=3,stride=2, bias=True, padding=1), 
            nn.CELU(),
            nn.Conv2d(128, 128, kernel_size=3,stride=2, bias=True, padding=1), 
            nn.CELU(),
            #nn.Conv2d(128, 128, kernel_size=3,stride=2, bias=True, padding=1), 
            #nn.CELU(),
            #nn.Conv2d(128, 128, kernel_size=3,stride=2, bias=True, padding=1), 
            #nn.CELU(),
            View(128 * 4 * 4),
            nn.CELU(), 
            nn.Linear(128 * 4 * 4, 128),
            nn.CELU(), 
            nn.Linear(128, 64),
            nn.CELU(), 
            nn.Linear(64, 32),
            nn.CELU(), 
            nn.Linear(32, 1),
            View()
        ).cuda()

    def forward(self, input): 
        output = self.first_linear(input) + self.first_squared(input) ** 2
        output = self.convex(output)
        return output
    
    def push(self, input):
        return autograd.grad(
            outputs=self.forward(input), inputs=input,
            create_graph=True, retain_graph=True,
            only_inputs=True, grad_outputs=torch.ones(input.size()[0]).cuda().float()
        )[0]
    
    def convexify(self):
        for layer in self.convex:
            if (isinstance(layer, nn.Linear)) or (isinstance(layer, nn.Conv2d)):
                layer.weight.data.clamp_(0)