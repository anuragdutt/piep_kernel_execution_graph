#!/usr/bin/env python3
"""Map each unique CUDA kernel to its source module/layer in the BLOOM model.

This script processes the PyTorch profiler trace (with layer annotations) and maps
each unique kernel to the transformer layer(s) that invoked it.

Usage:
    python map_kernels_to_layers.py trace_with_layers.json output.json
"""

import json
import sys
import re
from collections import defaultdict
from typing import Dict, List, Set


def extract_layer_from_stack(stack: str) -> str:
    """Extract the most specific layer annotation from call stack.
    
    Examples:
        'transformer.h.0.self_attention' -> 'transformer.h.0.self_attention'
        'transformer.h.23.mlp' -> 'transformer.h.23.mlp'
    """
    # Look for transformer.h.X patterns
    patterns = [
        r'transformer\.h\.(\d+)\.self_attention',
        r'transformer\.h\.(\d+)\.mlp',
        r'transformer\.h\.(\d+)\.input_layernorm',
        r'transformer\.h\.(\d+)\.post_attention_layernorm',
        r'transformer\.h\.(\d+)',
    ]
    
    for pattern in patterns:
        match = re.search(pattern, stack)
        if match:
            return match.group(0)
    
    return None


def main():
    if len(sys.argv) != 3:
        print(f"Usage: {sys.argv[0]} <trace_with_layers.json> <output.json>")
        sys.exit(1)
    
    trace_path = sys.argv[1]
    output_path = sys.argv[2]
    
    print(f"Loading trace from {trace_path}...")
    with open(trace_path, 'r') as f:
        data = json.load(f)
    
    events = data['traceEvents']
    print(f"Loaded {len(events)} trace events")
    
    # Step 1: Build timestamp ranges for user_annotation events (layer context)
    print("\nBuilding layer annotation time ranges...")
    layer_annotations = []  # List of (start_ts, end_ts, layer_name)
    
    for event in events:
        if event.get('cat') == 'user_annotation':
            name = event.get('name', '')
            if 'transformer.h.' in name:
                start_ts = event.get('ts')
                dur = event.get('dur', 0)
                end_ts = start_ts + dur
                layer_annotations.append((start_ts, end_ts, name))
    
    print(f"Found {len(layer_annotations)} layer annotations")
    
    # Step 2: Map CPU ops (with External IDs) to layers based on timestamp overlap
    print("\nMapping CPU ops to layers by timestamp...")
    ext_id_to_layer = {}
    
    for event in events:
        if event.get('cat') == 'cpu_op':
            ext_id = event.get('args', {}).get('External id')
            ts = event.get('ts')
            
            if ext_id and ts:
                # Find which layer annotations this CPU op falls within
                layers = set()
                for start_ts, end_ts, layer_name in layer_annotations:
                    if start_ts <= ts <= end_ts:
                        layers.add(layer_name)
                
                if layers:
                    if ext_id not in ext_id_to_layer:
                        ext_id_to_layer[ext_id] = set()
                    ext_id_to_layer[ext_id].update(layers)
    
    print(f"Mapped {len(ext_id_to_layer)} external IDs to layers")
    
    # Step 2: Find cuda_runtime events and their correlation IDs
    print("\nMapping cuda_runtime External IDs to correlation IDs...")
    ext_id_to_corr = {}
    
    for event in events:
        if event.get('cat') == 'cuda_runtime':
            ext_id = event.get('args', {}).get('External id')
            corr = event.get('args', {}).get('correlation')
            if ext_id and corr:
                ext_id_to_corr[ext_id] = corr
    
    print(f"Found {len(ext_id_to_corr)} cuda_runtime events")
    
    # Step 3: Build correlation -> layer mapping
    corr_id_to_layer = {}
    for ext_id, layers in ext_id_to_layer.items():
        if ext_id in ext_id_to_corr:
            corr = ext_id_to_corr[ext_id]
            if corr not in corr_id_to_layer:
                corr_id_to_layer[corr] = set()
            corr_id_to_layer[corr].update(layers)
    
    print(f"Mapped {len(corr_id_to_layer)} correlation IDs to layers")
    
    # Step 4: Map each unique kernel to layers
    print("\nMapping kernels to layers...")
    kernel_to_layers = defaultdict(set)
    kernel_to_invocations = defaultdict(int)
    
    for event in events:
        if event.get('cat') == 'Kernel':
            kernel_name = event.get('name', '')
            corr = event.get('args', {}).get('correlation')
            
            if corr in corr_id_to_layer:
                layers = corr_id_to_layer[corr]
                kernel_to_layers[kernel_name].update(layers)
                kernel_to_invocations[kernel_name] += 1
    
    print(f"Mapped {len(kernel_to_layers)} unique kernels to layers")
    
    # Step 5: Convert to serializable format and compute statistics
    result = []
    for kernel_name in sorted(kernel_to_layers.keys()):
        layers = sorted(list(kernel_to_layers[kernel_name]))
        invocations = kernel_to_invocations[kernel_name]
        
        # Extract layer numbers
        layer_nums = []
        for layer in layers:
            match = re.search(r'transformer\.h\.(\d+)', layer)
            if match:
                layer_nums.append(int(match.group(1)))
        
        result.append({
            'kernel_name': kernel_name,
            'layers': layers,
            'layer_count': len(layers),
            'layer_numbers': sorted(set(layer_nums)),
            'invocation_count': invocations
        })
    
    # Save result
    print(f"\nSaving results to {output_path}...")
    with open(output_path, 'w') as f:
        json.dump(result, f, indent=2)
    
    # Print summary statistics
    print("\n=== Summary ===")
    print(f"Total unique kernels: {len(result)}")
    
    kernels_by_layer_count = defaultdict(int)
    for item in result:
        kernels_by_layer_count[item['layer_count']] += 1
    
    print("\nKernels by number of layers they appear in:")
    for count in sorted(kernels_by_layer_count.keys()):
        print(f"  {count} layer(s): {kernels_by_layer_count[count]} kernels")
    
    # Show a few examples
    print("\n=== Sample kernel-to-layer mappings ===")
    for item in result[:5]:
        print(f"\nKernel: {item['kernel_name'][:80]}")
        print(f"  Appears in {item['layer_count']} layers: {item['layers'][:3]}")
        print(f"  Total invocations: {item['invocation_count']}")
    
    print(f"\nDone! Wrote {len(result)} kernel mappings to {output_path}")


if __name__ == '__main__':
    main()
