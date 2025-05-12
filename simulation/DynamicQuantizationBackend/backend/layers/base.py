import abc
import torch
from torch import nn
from functools import partial
from backend.quantizers import QuantMode, BaseQuantizer


class QLayer(nn.Module, abc.ABC):
    def __init__(self, layer:nn.Module, mode:QuantMode, **qparams):
        super().__init__() 

        self.name = qparams.get('full_name', self._get_name())
        # Weights and bias
        self.fp_weight = layer.weight
        if mode == QuantMode.ESTIMATE:
            qparams['weights'] = self.fp_weight
            qparams['estim_func'] = self.estimateOutDist
            qparams['name']=self.name
        self.fp_bias = layer.bias

        self.per_channel= qparams.get('per_channel')

        self.quantizer: BaseQuantizer = mode.quantizer(**qparams)

        self.qweight = self.quantizer.quantize(
                    self.fp_weight,
                    symmetric = True,
                    per_channel = True, 
                    channel_dim=0
        )

        self.debug = qparams.get('debug', False)
        
    def forward(self, x:torch.Tensor):
        # Quantize input
        qx = self.quantizer.quantize(x, symmetric=False, per_channel=False)
        # Manage zero_point
        sc = qx.scale[(...,) + (None,) * (qx.tensor.ndim - qx.scale.ndim)]
        zp = qx.zero_point[(...,) + (None,) * (qx.tensor.ndim - qx.zero_point.ndim)]
        qx.tensor = qx.tensor - zp

        # Perform the forward pass for the respective layer using quantized weights and quantized input
        forward = self._layer_forward()
        res32 = forward(qx.tensor, self.qweight.tensor, None, **self.forward_args)
        # Add bias scaled accordingly to input and weights scales
        if self.bias:
            qbias = torch.round(
                        self.fp_bias[(None,) + (...,) + (None,) * (x.ndim - self.fp_bias.ndim -1)] / 
                        (self.qweight.scale[(None,) + (...,) + (None,) * (x.ndim - self.qweight.scale.ndim - 1)] * sc)
            )
            res32 += qbias
        else:
            qbias=None

        # Requantize to dtype bit using different requantizer basing on quantizer initialization
        re_sc, re_zp = self.quantizer.reQuant_params(**{'res': res32, 'qx': qx.tensor, 'bias':qbias})
        res8 = self.quantizer.quantize(
                        res32, 
                        symmetric=False, 
                        per_channel=self.per_channel,
                        scale=re_sc, 
                        zero_point=re_zp, 
        )
        # Update scale
        res8.scale =  (
            sc * 
            self.qweight.scale[(None,) + (...,) + (None,) * (x.ndim - self.qweight.scale.ndim - 1)] * 
            res8.scale[(...,) + (None,) * (x.ndim - res8.scale.ndim)] 
        )
        res_deq = self.quantizer.dequantize(res8)
        
        if self.debug:
            res_fp = self.forward_fp(x)
        
            # Calculate the MSE between the original and quantized outputs.
            self.loss_fn = nn.MSELoss()
            mse = self.loss_fn(res_deq, res_fp)

            # Compute cosine similarity along each sample (feature dimension is 1, batch dimension is 0)
            cosine_sim = torch.nn.functional.cosine_similarity(res_deq.flatten(1), res_fp.flatten(1), dim=1)
            avg_cosine_sim = cosine_sim.mean().item()
            print(self.name)
            print("MSE:", mse.item())
            print("Average cosine similarity:", avg_cosine_sim)

        return res_deq
    
    def forward_fp(self, x):
        forward = self._layer_forward()
        return forward(x, self.fp_weight, self.fp_bias, **self.forward_args)
    
    @staticmethod
    @abc.abstractmethod
    def estimateOutDist(x, w_mean, w_std, **forward_args):
        pass

    @abc.abstractmethod
    def _layer_forward(self, x, w, **forward_args):
        pass

    def extra_repr(self):
        return self.quantizer.extra_repr()