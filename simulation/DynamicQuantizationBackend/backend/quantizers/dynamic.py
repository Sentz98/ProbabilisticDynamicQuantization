from backend.quantizers.base import BaseQuantizer

class DynamicQuantizer(BaseQuantizer):
    def __init__(self,  **params):
        super().__init__(**params)
        

    def reQuant_params(self, **params):

        res = params.get('res')
        if self.per_channel:
            data_dims = list(range(2, res.ndim))
        else:
            data_dims = list(range(1, res.ndim))
        
        tensor_min = res.amin(dim=data_dims, keepdim=True)
        tensor_max = res.amax(dim=data_dims, keepdim=True)
        
        return self.compute_qparams(tensor_min, tensor_max, per_channel=self.per_channel)
    
    def extra_repr(self):
        return 'DYNAMIC QUANTIZER | '
        
        