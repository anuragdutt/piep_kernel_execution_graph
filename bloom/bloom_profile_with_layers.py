#!/usr/bin/env python3
"""BLOOM-560M profiling with layer-level module annotations.

This version adds profiler record_function annotations for each transformer layer,
so we can map kernels back to specific layers like 'transformer.h.0.self_attention.query_key_value'.
"""

import argparse
import json
import sys
import torch
from torch.profiler import record_function
from transformers import AutoModelForCausalLM, AutoTokenizer


def annotate_bloom_modules(model: torch.nn.Module) -> None:
    """Add profiler record_function wrappers to BLOOM layers."""
    
    # Annotate each transformer block
    for layer_idx, layer in enumerate(model.transformer.h):
        original_forward = layer.forward
        
        def make_forward(idx, orig_forward):
            def wrapped_forward(*args, **kwargs):
                with record_function(f"transformer.h.{idx}"):
                    return orig_forward(*args, **kwargs)
            return wrapped_forward
        
        layer.forward = make_forward(layer_idx, original_forward)
        
        # Annotate attention sublayers
        attn = layer.self_attention
        original_attn_forward = attn.forward
        
        def make_attn_forward(idx, orig_forward):
            def wrapped_forward(*args, **kwargs):
                with record_function(f"transformer.h.{idx}.self_attention"):
                    return orig_forward(*args, **kwargs)
            return wrapped_forward
        
        attn.forward = make_attn_forward(layer_idx, original_attn_forward)
        
        # Annotate MLP
        mlp = layer.mlp
        original_mlp_forward = mlp.forward
        
        def make_mlp_forward(idx, orig_forward):
            def wrapped_forward(*args, **kwargs):
                with record_function(f"transformer.h.{idx}.mlp"):
                    return orig_forward(*args, **kwargs)
            return wrapped_forward
        
        mlp.forward = make_mlp_forward(layer_idx, original_mlp_forward)
        
        # Annotate layer norms
        ln1 = layer.input_layernorm
        ln2 = layer.post_attention_layernorm
        
        def make_ln_forward(idx, ln_name, orig_forward):
            def wrapped_forward(*args, **kwargs):
                with record_function(f"transformer.h.{idx}.{ln_name}"):
                    return orig_forward(*args, **kwargs)
            return wrapped_forward
        
        ln1.forward = make_ln_forward(layer_idx, "input_layernorm", ln1.forward)
        ln2.forward = make_ln_forward(layer_idx, "post_attention_layernorm", ln2.forward)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="bigscience/bloom-560m")
    parser.add_argument("--prompt", default="Explain machine learning")
    parser.add_argument("--max-new-tokens", type=int, default=5)
    parser.add_argument("--trace", default="trace_with_layers.json")
    args = parser.parse_args()
    
    device = torch.device("cuda:0")
    print(f"Loading model {args.model}...", file=sys.stderr)
    
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=False)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True
    )
    model.eval()
    model.to(device)
    
    # Add layer annotations
    print("Adding layer annotations...", file=sys.stderr)
    annotate_bloom_modules(model)
    
    inputs = tokenizer(args.prompt, return_tensors="pt").to(device)
    
    # Warmup
    print("Warmup...", file=sys.stderr)
    with torch.no_grad():
        _ = model.generate(**inputs, max_new_tokens=2)
    torch.cuda.synchronize()
    
    # Profile with layer context
    print("Profiling...", file=sys.stderr)
    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA
        ],
        record_shapes=True,
        with_stack=True,
    ) as prof:
        with torch.no_grad():
            _ = model.generate(**inputs, max_new_tokens=args.max_new_tokens)
        torch.cuda.synchronize()
    
    prof.export_chrome_trace(args.trace)
    print(f"Wrote {args.trace}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
