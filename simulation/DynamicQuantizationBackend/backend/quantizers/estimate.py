from backend.quantizers.base import BaseQuantizer

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import math

def plot_out(data, in_data, bias, weights, mean=None, std=None, bins=int(1e3), name=None, add_real_distrib=False):
    comp_mean, comp_std = None, None
    if mean is None and std is None:
        mean = data.mean()
        std = data.std()
    elif add_real_distrib:
        # For real distribution, ensure conversion to Python floats if needed
        comp_mean = data.mean().item() if hasattr(data.mean(), 'item') else data.mean()
        comp_std = data.std().item() if hasattr(data.std(), 'item') else data.std()

    # Convert provided mean and std to floats (if they aren't already)
    mean_val = mean.item() if hasattr(mean, 'item') else float(mean)
    std_val = std.item() if hasattr(std, 'item') else float(std)

    # Print minimum and maximum values of the data
    print("Data min:", data.min())
    print("Data max:", data.max())

    # Compute and print the percentage of values excluded based on sigma thresholds
    for k in [1, 2, 3]:
        lower_bound = mean_val - k * std_val
        upper_bound = mean_val + k * std_val
        # Convert data to numpy if needed for element-wise operations
        data_np = data.numpy() if hasattr(data, 'numpy') else np.array(data)
        excluded = ((data_np < lower_bound) | (data_np > upper_bound)).sum()
        percentage_excluded = 100 * excluded / data_np.size
        print(f"Percentage of values excluded outside {k} sigma: {percentage_excluded:.2f}%")

    # Print minimum and maximum values of the data
    print("Estim min (3sigma):", mean_val - 3 * std_val)
    print("Estim max (3sigma):", mean_val + 3 * std_val)

    if bias is not None:
        fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(20, 30))
        ax3.set_title('BIAS Distribution', fontsize=12)
        sns.histplot(bias.flatten(), stat="density", kde=True, ax=ax3,
             color='yellow', label='Weights')
    else:
        fig, (ax1, ax2, ax4) = plt.subplots(3, 1, figsize=(20, 30))

    plt.suptitle('Distribution Comparison', y=1.02, fontsize=14)
    ax1.set_title('Input Data Distribution', fontsize=12)
    ax2.set_title('Weights Distribution', fontsize=12)
    ax4.set_title('Output Distribution', fontsize=12)

    sns.histplot(in_data.flatten(), stat="density", kde=True, ax=ax1,
             color='red', label='Input')
    
    sns.histplot(weights.flatten(), stat="density", kde=True, ax=ax2,
             color='green', label='Weights')
    
    # Plot histogram and KDE
    sns.histplot(data.flatten(), bins=bins, stat="density", kde=True, ax=ax4,
                 color='blue', label='Output Distribution')

    # Normal Distribution based on the provided or computed mean and std
    n = np.linspace(mean_val - 3*std_val, mean_val + 3*std_val, int(1e3))
    pdf = (1 / (std_val * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((n - mean_val) / std_val)**2)
    ax4.plot(n, pdf, color='red', label='Normal Distribution')

    # Plot the best fit normal distribution if real distribution values are provided
    if comp_mean is not None and comp_std is not None:
        n_comp = np.linspace(comp_mean - 3*comp_std, comp_mean + 3*comp_std, int(1e3))
        pdf_comp = (1 / (comp_std * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((n_comp - comp_mean) / comp_std)**2)
        ax4.plot(n_comp, pdf_comp, color='green', label='Best Fit Normal Distribution')

    # Plot vertical lines for sigma markers
    for k in [1, 2, 3]:
        for sign in [-1, 1]:
            x_val = mean_val + sign * k * std_val
            pdf_val = (1 / (std_val * np.sqrt(2 * np.pi))) * np.exp(-0.5 * (k**2))
            ax4.axvline(x=x_val, color='green', linestyle='--', alpha=0.7, label=f'{k} sigma')
            ax4.plot(x_val, pdf_val, 'o', color='green')
    if name is not None:
        os.makedirs('imgs', exist_ok=True)
        plt.savefig(f'imgs/{name}.png')
        plt.close()
    else:
        plt.show()


class EstimateQuantizer(BaseQuantizer):
    def __init__(self, **params):
        super().__init__(**params)
        # Attributes for Estimate
        self.e_std = params.pop("e_std", None) 

        self.name = params.get('name')
        
        self.weights = self.quantize(
                    params.pop("weights", None),
                    symmetric = True,
                    per_channel = True,
                    channel_dim=0
        )
        
        if self.per_channel:
            dims = list(range(1, self.weights.tensor.ndim))
            self.w_mean = self.weights.tensor.mean(dims, keepdim=True).unsqueeze(0).squeeze(-1)
            self.w_std = self.weights.tensor.std(dims, keepdim=True).unsqueeze(0).squeeze(-1)
        else:
            self.w_mean = self.weights.tensor.mean(list(range(0, self.weights.tensor.ndim)), keepdim=True)
            self.w_std = self.weights.tensor.std(list(range(0, self.weights.tensor.ndim)), keepdim=True)
            

        self.forward_args_estim = params.pop("forward_args", {}).copy() 

        self.forward_args_estim['sampling_stride'] = params.pop("sampling_stride", None)
        self.forward_args_estim['kernel_size'] = params.pop("kernel_size", None)

        self.estim_func = params.pop("estim_func", None) 

        self.debug=params.pop('debug', False)
        

        self.alpha_l=None
        self.alpha_h=None
        self.calibrate=False
        self.calib_size=params.pop('calib', 16)
        self.calib = 0

    def reQuant_params(self, **params):
        qx = params.get('qx')
        bias = params.get('bias', None)
        res = params.get('res') 
        
        y_mean, y_std = self.estim_func(qx, bias, self.w_mean, self.w_std, per_channel=self.per_channel, **self.forward_args_estim)
        
        if not self.calibrate:
            if self.per_channel:
                dims = list(range(2, res.ndim))
                n_l = torch.round(
                    ((y_mean - res.amin(dims, keepdim=True)) / (y_std + torch.finfo(torch.float32).eps)).amax(list(range(1, res.ndim))).mean()
                )
                n_h = torch.round(
                    ((res.amax(dims, keepdim=True) - y_mean) / (y_std + torch.finfo(torch.float32).eps)).amax(list(range(1, res.ndim))).mean()
                )
            else:
                dims = list(range(1, res.ndim))
                n_l = torch.round(
                ((y_mean - (res.amin(dims, keepdim=True))) / (y_std + torch.finfo(torch.float32).eps)).mean()
                )
                n_h = torch.round(((res.amax(dims, keepdim=True) - y_mean) / (y_std + torch.finfo(torch.float32).eps)).mean())
            

            self.alpha_l = torch.round((self.alpha_l + n_l) / 2) if self.alpha_l is not None else n_l
            self.alpha_h = torch.round((self.alpha_h + n_h) / 2) if self.alpha_h is not None else n_h
            print(f"{self.name}: n_l {self.alpha_l}, n_h {self.alpha_h}")

            self.calib += res.shape[0]

            if self.calib>=self.calib_size:
                self.calibrate=True
        
        q_min = y_mean - self.alpha_l * y_std
        q_max = y_mean + self.alpha_h * y_std

        if self.debug:                      
            print('estimated')
            print(y_mean, y_std)
            print('computed')
            print(res.mean(), res.std())
            plot_out(res.detach(), qx.detach(), bias, self.weights.tensor.detach(), y_mean.detach().flatten().numpy(), y_std.detach().flatten().numpy(), name=self.name, add_real_distrib=True)

        
        # breakpoint()
        return self.compute_qparams(q_min, q_max, per_channel=self.per_channel)
    
    def extra_repr(self):
        return (f'ESTIMATE QUANTIZER, e_std:{self.e_std}'
                + (f", sampling_stride:{self.forward_args_estim['sampling_stride']}" if self.forward_args_estim['sampling_stride'] is not None else '')
                + ' | ')