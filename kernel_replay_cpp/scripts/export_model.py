#!/usr/bin/env python3
"""Export BLOOM-560M model to TorchScript for C++ inference.

This script loads the BLOOM-560M model from HuggingFace and exports it
to TorchScript format (.pt file) that can be loaded in C++ libtorch.

Usage:
    python export_model.py [--output bloom_560m_traced.pt]
"""

import argparse
import os
import torch
from transformers import AutoModelForCausalLM

def export_bloom_model(model_name: str, output_path: str, example_seq_len: int = 64, use_fp32: bool = False):
    """
    Export BLOOM model to TorchScript.
    
    Args:
        model_name: HuggingFace model identifier (e.g., 'bigscience/bloom-560m')
        output_path: Path to save the traced model (.pt file)
        example_seq_len: Sequence length for the example input used in tracing
        use_fp32: Use FP32 precision (CPU). Default: FP16 on GPU (matches profiling)
    """
    print(f"Loading model: {model_name}")
    
    # Default to FP16 on GPU to match the original bloom_profile.py
    if use_fp32:
        print("Using FP32 precision (CPU tracing)")
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float32,
            low_cpu_mem_usage=True
        )
        model.eval()
        model = model.cpu()
        device = torch.device("cpu")
    else:
        print("Using FP16 precision on GPU (matches original profiling)")
        if not torch.cuda.is_available():
            raise RuntimeError("GPU tracing requires CUDA, but no GPU found!")
        
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True
        )
        model.eval()
        
        print("Moving model to GPU for FP16 tracing...")
        model = model.cuda()
        device = torch.device("cuda")
    
    # Wrap model to only return logits (TorchScript can't handle complex nested structures)
    class ModelWrapper(torch.nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model
        
        def forward(self, input_ids):
            # Only return logits, not past_key_values or other complex outputs
            outputs = self.model(input_ids, use_cache=False)
            return outputs.logits
    
    wrapped_model = ModelWrapper(model)
    wrapped_model.eval()
    
    # Create example input (batch=1, seq_len=example_seq_len)
    print(f"Creating example input (seq_len={example_seq_len})...")
    example_input = torch.randint(0, 50000, (1, example_seq_len), dtype=torch.long, device=device)
    
    print("Tracing wrapped model (logits only) with torch.jit.trace...")
    traced_model = torch.jit.trace(wrapped_model, example_input)
    
    print(f"Saving traced model to {output_path}...")
    traced_model.save(output_path)
    
    # Get file size
    file_size_gb = os.path.getsize(output_path) / (1024**3)
    print(f"✓ Model exported successfully!")
    print(f"  Output: {output_path}")
    print(f"  Size: {file_size_gb:.2f} GB")
    
    # Verify we can load it back
    print("Verifying export...")
    loaded = torch.jit.load(output_path)
    test_output = loaded(example_input)
    print(f"✓ Verification successful! Output shape: {test_output.shape}")

def main():
    parser = argparse.ArgumentParser(description="Export BLOOM model to TorchScript")
    parser.add_argument(
        "--model",
        default="bigscience/bloom-560m",
        help="HuggingFace model name (default: bigscience/bloom-560m)"
    )
    parser.add_argument(
        "--output",
        default="bloom_560m_traced.pt",
        help="Output path for traced model (default: bloom_560m_traced.pt)"
    )
    parser.add_argument(
        "--seq-len",
        type=int,
        default=64,
        help="Sequence length for example input (default: 64)"
    )
    parser.add_argument(
        "--fp32",
        action="store_true",
        help="Use FP32 on CPU instead of FP16 on GPU (default: FP16 on GPU)"
    )
    
    args = parser.parse_args()
    
    export_bloom_model(args.model, args.output, args.seq_len, use_fp32=args.fp32)

if __name__ == "__main__":
    main()
