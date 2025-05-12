import sys
import os
import unittest
import argparse
import torch
import torch.nn as nn
from backend import QLinear, QuantMode
import random

# Ensure the parent directory is in the path for module discovery.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

class TestQuantizedLinear(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Set up argument parser to allow configuration via command-line.
        parser = argparse.ArgumentParser(description="Test Quantized Linear Layer")
        parser.add_argument("--tolerance", type=float, default=5e-1, help="Tolerable MSE threshold")
        parser.add_argument("--size", type=int, nargs=2, default=[20, 1280],
                            help="Default size of the input tensor (N, Features)")
        parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose logging")
        args, _ = parser.parse_known_args()

        # Store parsed arguments in class variables.
        cls.tolerance = args.tolerance
        cls.size = args.size
        cls.verbose = args.verbose

    def setUp(self):
        # Create and initialize the original Linear layer using the default shape.
        self.linear_layer = nn.Linear(in_features=self.size[1], out_features=64)
        nn.init.normal_(self.linear_layer.weight, mean=0.0, std=0.1)  # Standard random initialization.

        # Define a sample input tensor using the default shape.
        self.input_tensor = torch.randn(size=self.size)

        # Compute the original output for later comparison.
        self.original_output = self.linear_layer(self.input_tensor)

        # Loss function to compute the error between original and quantized outputs.
        self.loss_fn = nn.MSELoss()
        self.num_runs = 5  # Number of iterations per shape.
        self.shapes = [
            (2, 1280),
            (4, 1024),
            (8, 512),
            (16, 256)
        ]

    def test_dynamic(self):
        # Instantiate the quantized linear layer using dynamic quantization.
        quantized_linear = QLinear(self.linear_layer, QuantMode.DYNAMIC)

        # Compute the quantized output.
        quantized_output = quantized_linear(self.input_tensor)

        # Calculate the MSE between the original and quantized outputs.
        mse = self.loss_fn(self.original_output, quantized_output)

        # Compute cosine similarity along each sample (feature dimension is 1, batch dimension is 0)
        cosine_sim = torch.nn.functional.cosine_similarity(self.original_output, quantized_output, dim=1)
        avg_cosine_sim = cosine_sim.mean().item()

        if self.verbose:
            print('DYNAMIC-------------------------------------------------------')
            print("MSE:", mse.item())
            print("Cosine similarity per sample:", cosine_sim)
            print("Average cosine similarity:", avg_cosine_sim)

        # Check that the error does not exceed the defined tolerance.
        self.assertLessEqual(
            mse.item(), self.tolerance,
            f"Quantization error {mse.item()} exceeds tolerance {self.tolerance}"
        )

        # Define a threshold for cosine similarity (e.g., 0.95) to ensure directional consistency.
        cosine_threshold = 0.95
        self.assertGreaterEqual(
            avg_cosine_sim, cosine_threshold,
            f"Average cosine similarity {avg_cosine_sim} is below threshold {cosine_threshold}"
        ) 
    
    
    def test_estimate(self):
        # Instantiate the quantized linear layer using dynamic quantization.
        quantized_linear = QLinear(self.linear_layer, QuantMode.ESTIMATE, e_std=3)

        # Compute the quantized output.
        quantized_output = quantized_linear(self.input_tensor)

        # Calculate the MSE between the original and quantized outputs.
        mse = self.loss_fn(self.original_output, quantized_output)

        # Compute cosine similarity along each sample (feature dimension is 1, batch dimension is 0)
        cosine_sim = torch.nn.functional.cosine_similarity(self.original_output, quantized_output, dim=1)
        avg_cosine_sim = cosine_sim.mean().item()

        if self.verbose:

            print('ESTIMATE-------------------------------------------------------')
            print("MSE:", mse.item())
            print("Cosine similarity per sample:", cosine_sim)
            print("Average cosine similarity:", avg_cosine_sim)

        # Check that the error does not exceed the defined tolerance.
        # self.assertLessEqual(
        #     mse.item(), self.tolerance,
        #     f"Quantization error {mse.item()} exceeds tolerance {self.tolerance}"
        # )

        # Define a threshold for cosine similarity (e.g., 0.95) to ensure directional consistency.
        cosine_threshold = 0.95
        self.assertGreaterEqual(
            avg_cosine_sim, cosine_threshold,
            f"Average cosine similarity {avg_cosine_sim} is below threshold {cosine_threshold}"
        )


    def test_linear(self):
        # Define a set of diverse input shapes (batch size, number of features).
        
        

        mse_list = []
        mean_diff_list = []
        std_diff_list = []

        for shape in self.shapes:
            for _ in range(self.num_runs):
                # Randomize weight initialization parameters.
                weight_mean = random.uniform(-0.5, 0.5)
                weight_std = random.uniform(0.01, 0.2)

                # Create a new linear layer with the current shape's feature size.
                linear_layer = nn.Linear(in_features=shape[1], out_features=64)
                nn.init.normal_(linear_layer.weight, mean=weight_mean, std=weight_std)
                # Optionally zero out bias for consistency.
                if linear_layer.bias is not None:
                    nn.init.constant_(linear_layer.bias, 0.0)

                # Generate a random input tensor for the current shape.
                input_tensor = torch.randn(shape)

                # Compute the original output.
                original_output = linear_layer(input_tensor)

                # Instantiate the quantized linear layer.
                quantized_linear = QLinear(linear_layer, QuantMode.ESTIMATE, e_std=2)
                quantized_output = quantized_linear(input_tensor)

                # Compute MSE between outputs.
                mse = self.loss_fn(original_output, quantized_output).item()
                mse_list.append(mse)

                # Compute the difference in output means.
                orig_mean = original_output.mean().item()
                quant_mean = quantized_output.mean().item()
                mean_diff = abs(orig_mean - quant_mean)
                mean_diff_list.append(mean_diff)

                # Compute the difference in output standard deviations.
                orig_std = original_output.std().item()
                quant_std = quantized_output.std().item()
                std_diff = abs(orig_std - quant_std)
                std_diff_list.append(std_diff)

        # Calculate average statistics over all runs.
        avg_mse = sum(mse_list) / len(mse_list)
        avg_mean_diff = sum(mean_diff_list) / len(mean_diff_list)
        avg_std_diff = sum(std_diff_list) / len(std_diff_list)

        if self.verbose:
            print("Extensive Quantization Statistics:")
            print("Average MSE:", avg_mse)
            print("Average mean difference:", avg_mean_diff)
            print("Average std difference:", avg_std_diff)


if __name__ == "__main__":
    unittest.main()
