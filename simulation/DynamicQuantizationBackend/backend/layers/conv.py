import torch 
import torch.nn as nn
from backend.layers.base import QLayer
from backend.quantizers import QTensor, QuantMode

def _to_tuple(param):
    return param if isinstance(param, tuple) else (param, param)

class QConv2d(QLayer):
    def __init__(self, conv_layer, mode:QuantMode, **qparams):

        # Optionally get sampling_stride from qparams (defaulting to 1) and convert it too.
        sampling_stride = qparams.get("sampling_stride", 1)
        qparams['sampling_stride'] = _to_tuple(sampling_stride)

        # Extract parameters, converting them to tuples
        self.in_channels = conv_layer.in_channels
        self.out_channels = conv_layer.out_channels
        self.kernel_size = _to_tuple(conv_layer.kernel_size)
        self.stride = _to_tuple(conv_layer.stride)
        self.padding = _to_tuple(conv_layer.padding)
        self.dilation = _to_tuple(conv_layer.dilation)
        self.groups = conv_layer.groups
        self.bias = conv_layer.bias is not None
        self.padding_mode = conv_layer.padding_mode

        self.forward_args = {
            'stride': _to_tuple(conv_layer.stride),
            'padding': _to_tuple(conv_layer.padding),
            'dilation': _to_tuple(conv_layer.dilation),
            'groups': conv_layer.groups
        }
        qparams['forward_args'] = self.forward_args

        qparams['kernel_size'] = self.kernel_size

        super().__init__(conv_layer, mode, **qparams)
    
    def _layer_forward(self):
        return nn.functional.conv2d

    @staticmethod
    def estimateOutDist(x, bias, w_mean, w_std, per_channel=False,  **forward_args):
        if bias is None:
            bias = torch.zeros((1,1,1,1)).to(x.device)
        # Unpack parameters for height and width
        pad_h, pad_w = forward_args.get('padding')
        stride_h, stride_w = forward_args.get('stride')
        dil_h, dil_w = forward_args.get('dilation')
        samp_h, samp_w = forward_args.get('sampling_stride')

        # Pad the input (note: nn.functional.pad expects (pad_left, pad_right, pad_top, pad_bottom))
        padded_x = nn.functional.pad(x, (pad_w, pad_w, pad_h, pad_h))
        _, _, x_padded_h, x_padded_w = padded_x.shape
        k_h, k_w = forward_args.get('kernel_size')

        # Calculate output dimensions using the unpacked parameters
        original_output_h = ((x_padded_h - dil_h * (k_h - 1)) - 1) // stride_h + 1
        original_output_w = ((x_padded_w - dil_w * (k_w - 1)) - 1) // stride_w + 1
        output_height = ((original_output_h - 1) // samp_h) + 1
        output_width = ((original_output_w - 1) // samp_w) + 1

        y_mean = []
        y_std = []

        for i in range(output_height):
            for j in range(output_width):
                start_h = i * samp_h * stride_h
                start_w = j * samp_w * stride_w
                end_h = start_h + (k_h - 1) * dil_h + 1
                end_w = start_w + (k_w - 1) * dil_w + 1

                # Extract the patch using the corresponding dilation per dimension
                patch = padded_x[
                    :, :,
                    start_h:end_h:dil_h,
                    start_w:end_w:dil_w
                ]
                if per_channel:
                    y_mean.append((w_mean * patch.sum(dim=(1, 2, 3), keepdim=True)) + bias)
                    y_std.append(
                        torch.sqrt(
                            ((w_std**2) * (patch**2).sum((1, 2, 3), keepdim=True)) )
                        )
                    
                else:
                    y_mean.append(w_mean * patch.sum(dim=(1, 2, 3), keepdim=True) + bias.mean(dim=(1, 2, 3), keepdim=True))
                    y_std.append(torch.sqrt(
                            ((w_std**2) * (patch**2).sum((1, 2, 3), keepdim=True))) )
                    

        mus = torch.stack(y_mean, dim=0)
        sigmas = torch.stack(y_std, dim=0)
        combined_mu = torch.mean(mus, dim=0, keepdim=True) # Compute the combined mean
        sum_var = torch.sum(sigmas ** 2, dim=0) # Sum of variances from each distribution
        sum_sq_diff = torch.sum((mus - combined_mu) ** 2, dim=0) # Sum of squared differences between individual means and the combined mean
        N = mus.size(0) # Number of distributions
        return ( 
            mus.mean(dim=0), 
            torch.sqrt((sum_var + sum_sq_diff) / N)
        )

    
    def extra_repr(self):
        # Use extra info from BaseQuantizer if available.
        q_info = super().extra_repr() if hasattr(super(), "extra_repr") else ""
        
        s = (
            f"{q_info}{self.in_channels}, {self.out_channels}, "
            f"kernel_size={self.kernel_size}, "
            f"stride={self.forward_args['stride']}"
        )
        
        padding = self.forward_args.get("padding")
        if padding and padding != (0,) * len(padding):
            s += f", padding={padding}"
        
        dilation = self.forward_args.get("dilation")
        if dilation and dilation != (1,) * len(dilation):
            s += f", dilation={dilation}"
        
        groups = self.forward_args.get("groups", 1)
        if groups != 1:
            s += f", groups={groups}"
        
        if not self.bias:
            s += ", bias=False"
            
        padding_mode = self.padding_mode
        if padding_mode != "zeros":
            s += f", padding_mode={padding_mode}" #IDK what happen to functional i think doesn't work
        
        return s
