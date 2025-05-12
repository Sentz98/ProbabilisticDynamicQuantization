import abc
import torch
from backend.quantizers.qtensor import QTensor 

from torch.ao.quantization.utils import determine_qparams


class BaseQuantizer(abc.ABC):
    def __init__(self, **params):

        self.per_channel = params.get('per_channel', True)
        
        # General Quantization Parameters
        self.dtype = params.pop("dtype", torch.int8) 
        if not self.dtype.is_floating_point:
            self.dtype_info = torch.iinfo(self.dtype)
        else:
            self.dtype_info = torch.finfo(self.dtype)
    
    
    def quantize(
            self, 
            x:torch.Tensor,
            symmetric:bool, 
            per_channel:bool, 
            scale:torch.Tensor = None,
            zero_point:torch.Tensor = None,
            channel_dim:int=1) -> QTensor:
        
        if per_channel:
            q_dim = channel_dim + 1
        else:
            q_dim = channel_dim
        data_dims = list(range(q_dim, x.ndim))
        
        if scale is None or zero_point is None:
            tensor_min = x.amin(dim=data_dims, keepdim=True)
            tensor_max = x.amax(dim=data_dims, keepdim=True)

            if symmetric:
                if per_channel:
                    qscheme = torch.per_channel_symmetric
                else:
                    qscheme = torch.per_tensor_symmetric
            else:
                if per_channel:
                    qscheme=torch.per_channel_affine
                else:
                    qscheme=torch.per_tensor_affine

            scale, zero_point =determine_qparams(  
                        tensor_min, 
                        tensor_max, 
                        self.dtype_info.min, 
                        self.dtype_info.max,
                        self.dtype,
                        torch.tensor([torch.finfo(torch.float32).eps], device=tensor_min.device), 
                        has_customized_qrange=False,
                        qscheme=qscheme
            )
        quantized_tensor = torch.clamp(
            torch.round(x /scale) + zero_point, self.dtype_info.min, self.dtype_info.max
        )
            
        if quantized_tensor.isnan().any():
                breakpoint()
                
        return QTensor(
            quantized_tensor, scale.squeeze(data_dims), zero_point.squeeze(data_dims)
        )
    
    def dequantize(self, x:QTensor) -> torch.Tensor:
        if not isinstance(x, QTensor):
            raise ValueError(
                "Input must be a Qtensor object with tensor, scale, and zero_point attributes."
            )

        sc = x.scale[(...,) + (None,) * (x.tensor.ndim - x.scale.ndim)]
        zp = x.zero_point[(...,) + (None,) * (x.tensor.ndim - x.zero_point.ndim)]
        # Perform dequantization
        dequantized_tensor = ((x.tensor - zp) * sc).to(torch.float32)

        if dequantized_tensor.isnan().any():
                breakpoint()
        return dequantized_tensor
    
    @abc.abstractmethod
    def reQuant_params(self, **params):
        pass
    
    def compute_qparams(self, qmin, qmax, per_channel):
        if per_channel:
            qscheme = torch.per_channel_affine
        else:
            qscheme = torch.per_tensor_affine
        return determine_qparams(  
                        qmin, 
                        qmax, 
                        self.dtype_info.min, 
                        self.dtype_info.max,
                        self.dtype,
                        torch.tensor([torch.finfo(torch.float32).eps], device=qmin.device), 
                        has_customized_qrange=False,
                        qscheme= qscheme
        )

    @abc.abstractmethod 
    def extra_repr(self) -> str:
        pass
