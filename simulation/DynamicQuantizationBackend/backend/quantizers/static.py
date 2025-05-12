import torch
from backend.quantizers.base import BaseQuantizer

class StaticQuantizer(BaseQuantizer):
    def __init__(self, **params):
        super().__init__(**params)
        # Attributes for Static
        self.scale = None
        self.zero_point = None

        self.calibration_size = params.get('cal_size')
        self.calibration_set = None

        self.calibrated = False

    def reQuant_params(self, **params):
        if not self.calibrated:
            res = params.get('res')
            if self.per_channel:
                data_dims = list(range(2, res.ndim))
            else:
                data_dims = list(range(1, res.ndim))

            if self.calibration_set is None:
                self.calibration_set = [
                    (res[i].unsqueeze(0).amin(dim=data_dims, keepdim=True), res[i].unsqueeze(0).amax(dim=data_dims, keepdim=True))
                    for i in range(res.shape[0])
                ]  # Store amin, amax pairs for each image
            else:
                self.calibration_set.extend([
                    (res[i].unsqueeze(0).amin(dim=data_dims, keepdim=True), res[i].unsqueeze(0).amax(dim=data_dims, keepdim=True))
                    for i in range(res.shape[0])
                ])  # Append min/max pairs individually

            if len(self.calibration_set) >= self.calibration_size:
                print("!!! Calibration ---------------")

                # Unzip min and max values from the stored list
                min_tensors, max_tensors = zip(*self.calibration_set[:self.calibration_size])
                # Stack tensors to compute final min/max statistics
                tensor_min = torch.cat(min_tensors, dim=0).amin(dim=0, keepdim=True)
                tensor_max = torch.cat(max_tensors, dim=0).amax(dim=0, keepdim=True)

                # self.calibration_set = []
                self.calibrated = True

                self.scale, self.zero_point = self.compute_qparams(
                    tensor_min, 
                    tensor_max, 
                    per_channel=self.per_channel
                )
                # print(f"COMPUTED PARAMS= {self.scale}, {self.zero_point}")
                # breakpoint()
                return self.scale, self.zero_point
            
            else:
                print("!!! Creating Calibration Set")

                tensor_min = res.amin(dim=data_dims, keepdim=True)
                tensor_max = res.amax(dim=data_dims, keepdim=True)
                
                return self.compute_qparams(tensor_min, tensor_max, per_channel=self.per_channel)
        
        else:
            return self.scale, self.zero_point 

    def extra_repr(self):
        return 'STATIC QUANTIZER | '
