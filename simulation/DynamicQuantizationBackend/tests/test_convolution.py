import sys
import os
import unittest
import argparse
import torch
import torch.nn as nn
from backend import QConv2d, QuantMode
import random

# Ensure the parent directory is in the path for module discovery.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

class NormalizedMSELoss(torch.nn.Module):
    def __init__(self):
        super(NormalizedMSELoss, self).__init__()

    def forward(self, float_tensor, dequantized_tensor):
        # Compute the squared error
        squared_error = torch.sum((float_tensor - dequantized_tensor) ** 2)
        # Compute the squared norm of the original tensor
        norm = torch.sum(float_tensor ** 2)
        # Avoid division by zero
        if norm == 0:
            raise ValueError("The norm of the float tensor is zero. Cannot compute NMSE.")
        # Return the normalized MSE
        return squared_error / norm

class TestQuantizedConv2d(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Set up argument parser
        parser = argparse.ArgumentParser(description="Test Quantization")
        parser.add_argument("--tolerance", type=float, default=5e-1, help="Tolerable MSE threshold")
        parser.add_argument("--size", type=int, nargs=4, default=[2, 3, 32, 32], help="Size of the input tensor (N, C, H, W)")
        parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose logging")
        args, _ = parser.parse_known_args()

        # Parse arguments and set class variables
        cls.tolerance = args.tolerance
        cls.size = args.size
        cls.verbose = args.verbose

    def setUp(self):
        # Create and initialize the original Conv2d layer
        self.conv_layer = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1)
        nn.init.normal_(self.conv_layer.weight, mean=300, std=250)  # Random initialization for weights

        # Define a sample input tensor
        min_val = -120 # Minimum value of the range
        max_val = 122    # Maximum value of the range

        # Generate a tensor with values uniformly distributed within the range
        self.input_tensor = torch.rand(self.size) * (max_val - min_val) + min_val
        # self.input_tensor = torch.randn(size=self.size)  # Random input tensor

        # Calculate original output for comparison
        self.original_output = self.conv_layer(self.input_tensor)

        # if self.verbose:
        #     print("original")
        #     print(self.original_output)

        # Define a loss function for comparison
        self.loss_fn = NormalizedMSELoss()

        self.num_runs = 5  # Number of iterations per shape.
        self.shapes = [
            (2, 1280),
            (4, 1024),
            (8, 512),
            (16, 256)
        ]
        

    def test_dynamic(self):
        # Instantiate the quantized linear layer using dynamic quantization.
        quantized_linear = QConv2d(self.conv_layer, QuantMode.DYNAMIC)

        # Compute the quantized output.
        quantized_output = quantized_linear(self.input_tensor)

        # Calculate the MSE between the original and quantized outputs.
        mse = self.loss_fn(self.original_output, quantized_output)

        # Compute cosine similarity along each sample (feature dimension is 1, batch dimension is 0)
        cosine_sim = torch.nn.functional.cosine_similarity(self.original_output.flatten(1,3), quantized_output.flatten(1,3), dim=1)
        avg_cosine_sim = cosine_sim.mean().item()

        if self.verbose:
            print()
            print('DYNAMIC-------------------------------------------------------')
            print("MSE:", mse.item())
            print("Cosine similarity per sample:", cosine_sim)
            print("Average cosine similarity:", avg_cosine_sim)

        # Define a threshold for cosine similarity (e.g., 0.95) to ensure directional consistency.
        cosine_threshold = 0.95
        self.assertGreaterEqual(
            avg_cosine_sim, cosine_threshold,
            f"Average cosine similarity {avg_cosine_sim} is below threshold {cosine_threshold}"
        )
    
    
    def test_estimate(self):
        # Instantiate the quantized linear layer using dynamic quantization.
        quantized_linear = QConv2d(self.conv_layer, QuantMode.ESTIMATE, e_std=3)

        # Compute the quantized output.
        quantized_output = quantized_linear(self.input_tensor)

        # Calculate the MSE between the original and quantized outputs.
        mse = self.loss_fn(self.original_output, quantized_output)

        # Compute cosine similarity along each sample (feature dimension is 1, batch dimension is 0)
        cosine_sim = torch.nn.functional.cosine_similarity(self.original_output.flatten(1, 3), quantized_output.flatten(1,3), dim=1)
        avg_cosine_sim = cosine_sim.mean().item()

        if self.verbose:
            print()
            print('ESTIMATE-------------------------------------------------------')
            print("MSE:", mse.item())
            print("Cosine similarity per sample:", cosine_sim)
            print("Average cosine similarity:", avg_cosine_sim)


        # Define a threshold for cosine similarity (e.g., 0.95) to ensure directional consistency.
        cosine_threshold = 0.95
        self.assertGreaterEqual(
            avg_cosine_sim, cosine_threshold,
            f"Average cosine similarity {avg_cosine_sim} is below threshold {cosine_threshold}"
        )


if __name__ == "__main__":
    unittest.main()
