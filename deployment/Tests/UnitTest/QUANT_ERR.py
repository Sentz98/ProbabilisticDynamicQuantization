import math
import torch 
import re
import os

# --------------------I/O-------------------------------
def parse_logs(log_file):
    multipliers = []
    shifts = []
    outputs = []
    int32values = []
    with open(log_file, 'r') as f:
        for line in f:
            if line.startswith("MULTIPLIER:"):
                multipliers = list(map(int, line.split(":")[1].strip().split(',')))
            elif line.startswith("SHIFT:"):
                shifts = list(map(int, line.split(":")[1].strip().split(',')))
            elif line.startswith("OUTPUT:"):
                outputs = list(map(int, line.split(":")[1].strip().split(',')))
            elif line.startswith("ACCUMULATOR"):
                int32values.append(list(map(int, line.split(":")[1].strip().split(','))))
    return (
        torch.tensor(multipliers, dtype=torch.int32),     
        torch.tensor(shifts, dtype=torch.int8),           
        torch.tensor(outputs, dtype=torch.int8), 
        torch.tensor(int32values, dtype=torch.int32)         
    )

def load_pregenerated(folder_path):
    # Load scales
    scales = {}
    with open(f"{folder_path}/scales.txt", "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith("# Input_scale"):
                scales["input_scale"] = torch.tensor(
                    [float(next(f).strip())], 
                    dtype=torch.float32
                )
                
            elif line.startswith("# Input_zp"):
                scales["input_zp"] = torch.tensor(
                    [int(float(next(f).strip()))],  # Handle float-formatted integers
                    dtype=torch.int32
                )
                
            elif line.startswith("# Weight_scale"):
                scales["weight_scale"] = torch.tensor(
                    list(map(float, next(f).strip().split(","))),
                    dtype=torch.float32
                )
                
            elif line.startswith("# Output_scale"):
                scales["output_scale"] = torch.tensor(
                    [float(next(f).strip())],
                    dtype=torch.float32
                )
                
            elif line.startswith("# Output_zp"):
                scales["output_zp"] = torch.tensor(
                    [int(float(next(f).strip()))],  # Handle float-formatted integers
                    dtype=torch.int32
                )

    # Load output tensor
    with open(f"{folder_path}/output.txt", "r") as f:
        # Read shape from first line
        shape = tuple(map(int, next(f).strip().lstrip("# ").split(",")))
        
        # Read all values
        values = []
        for line in f:
            values.extend(map(float, line.strip().split(",")))
            
        output_tensor = torch.tensor(values, dtype=torch.float32).view(shape)

    return scales, output_tensor

def extract_tensor_from_file(file_path, dtype=torch.int32):
    """Reads a header file, extracts the first int32_t array, and converts it to a PyTorch tensor."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    with open(file_path, "r") as file:
        content = file.read()

    # General regex pattern to match any int32_t array, allowing for multiline declarations
    match = re.search(r"const int32_t \w+\[\d+\]\s*=\s*\{([^}]+)\};", content, re.DOTALL)
    if not match:
        raise ValueError(f"Array not found in: {file_path}")

    # Convert extracted values into a tensor
    values = list(map(int, re.sub(r'\s+', '', match.group(1)).split(",")))
    return torch.tensor(values, dtype=dtype)

def load_test_data(folder_path):
    """Loads and returns two tensors from header files in the given folder."""
    mult_tensor = extract_tensor_from_file(os.path.join(folder_path, "output_mult_data.h"))
    shift_tensor = extract_tensor_from_file(os.path.join(folder_path, "output_shift_data.h"))
    return mult_tensor, shift_tensor
#------------------------UTILS-----------------------------------------
def quantize_scale(scale):
    """Quantizes a scale value or a 1D tensor using Q31 representation."""
    significand, shift = torch.frexp(scale)  # Element-wise frexp
    significand_q31 = torch.round(significand * (1 << 31)).to(torch.int32)
    return significand_q31, shift.to(torch.int32)  # Ensure integer shift

def dequantize_scale(significand_q31, shift):
    """Dequantizes Q31 values back to floating point scale."""
    significand = significand_q31.to(torch.float32) / (1 << 31)
    scale = significand * torch.pow(2, shift.to(torch.float32))
    return scale

def print_tensor_info(name, tensor):
    """Helper function to print tensor information."""
    print(f"{name}: {tensor}")

def calculate_metrics(original_tensor, reconstructed_tensor, approach_name):
    """Calculate and print various metrics for comparing original and reconstructed tensors."""
    print(f"-------------------- {approach_name} Approach --------------------")
    
    # Mean Squared Error (MSE)
    mse = torch.mean((original_tensor - reconstructed_tensor) ** 2)
    print(f"MSE: {mse.item()}")
    
    # Mean Absolute Error (MAE)
    mae = torch.mean(torch.abs(original_tensor - reconstructed_tensor))
    print(f"MAE: {mae.item()}")
    
    # Signal-to-Quantization Noise Ratio (SQNR)
    signal_power = torch.sum(original_tensor ** 2)
    noise_power = torch.sum((original_tensor - reconstructed_tensor) ** 2)
    sqnr = 10 * torch.log10(signal_power / noise_power)
    print(f"SQNR (dB): {sqnr.item()}")
    
    # Peak Signal-to-Noise Ratio (PSNR)
    max_value = torch.max(original_tensor)
    psnr = 20 * torch.log10(max_value / torch.sqrt(mse))
    print(f"PSNR (dB): {psnr.item()}")
    
    # Cosine Similarity
    cosine_sim = torch.nn.functional.cosine_similarity(original_tensor, reconstructed_tensor, dim=1)
    mean_cosine_sim = torch.mean(cosine_sim)
    print(f"Mean Cosine Similarity: {mean_cosine_sim.item()}")
    
    # Relative Error
    relative_error = torch.norm(original_tensor - reconstructed_tensor) / torch.norm(original_tensor)
    print(f"Relative Error: {relative_error.item()}")
    
    print("------------------------------------------------------------")

# -------------------------MAIN-------------------------------
if __name__ == "__main__":
    est_mult, est_shift, est_output, accumulators = parse_logs("build-cortex-m3-gcc/test_arm_convolve_dynamic_s8_output.log")

    scales, output_tensor = load_pregenerated('PregeneratedData/dynamic')

    ref_mult, ref_shift = load_test_data("TestCases/TestData/dynamic")

    est_scale = dequantize_scale(est_mult,est_shift)
    ref_scale = dequantize_scale(ref_mult,ref_shift)
    print(f"ESTIMATION = {est_scale} -> {(255 / est_scale).to(torch.int)} Range")
    print(f"REFERENCE (non proprio)= {ref_scale} -> {(255 / ref_scale).to(torch.int)} Range")

    
    scale2fp_dyn = (scales['input_scale'] * scales['weight_scale']) / est_scale
    scale2fp_stat = (scales['input_scale'] * scales['weight_scale']) / ref_scale
    
    original_tensor = output_tensor
    est_output = est_output.view(original_tensor.shape)
    scale2fp_dyn = scale2fp_dyn.view(*([1] * (est_output.dim() - 1)), scale2fp_dyn.size(0))
    dynamic_rec_tensor = ((est_output + scales['output_zp']) * scale2fp_dyn).view(original_tensor.shape)

    scale2fp_stat = scale2fp_stat.view(*([1] * (est_output.dim() - 1)), scale2fp_stat.size(0))
    static_rec_tensor = ((est_output + scales['output_zp']) * scale2fp_stat).view(original_tensor.shape)

    print_tensor_info("Original Tensor", original_tensor)
    print_tensor_info("Dynamic Reconstructed Tensor", dynamic_rec_tensor)
    print_tensor_info("Static Reconstructed Tensor", static_rec_tensor)

    # Calculate metrics for dynamic approach
    calculate_metrics(original_tensor, dynamic_rec_tensor, "Dynamic")

    # Calculate metrics for static approach
    calculate_metrics(original_tensor, static_rec_tensor, "Static")

    


    # breakpoint()