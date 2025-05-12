import torch 
import torch.nn as nn
from backend.layers.base import QLayer
from backend.quantizers import QuantMode

class QLinear(QLayer):
    def __init__(self, linear_layer, mode:QuantMode, **qparams):
        super().__init__(linear_layer, mode, **qparams)
        
        # Extracted parameters from the original layer
        self.in_features = linear_layer.in_features
        self.out_features = linear_layer.out_features
        self.bias = linear_layer.bias is not None

        self.forward_args = {}

    def _layer_forward(self):
        return nn.functional.linear
    
    @staticmethod
    def estimateOutDist(x:torch.Tensor, bias, w_mean, w_std, per_channel=False, **forward_args):
        if bias is None:
            bias = torch.zeros((1,1))

        if per_channel:
            return w_mean * x.sum(1, keepdim=True)+bias, torch.sqrt(w_std**2 * (x**2).sum(1, keepdim=True))
        else:
            return w_mean * x.sum(1, keepdim=True)+bias.mean(1, keepdim=True), w_std * torch.sqrt((x**2).sum(1, keepdim=True))
    
    def extra_repr(self) -> str:
        q_info = super().extra_repr()
        return f"{q_info}in_features={self.in_features}, out_features={self.out_features}, bias={self.bias is not None}"