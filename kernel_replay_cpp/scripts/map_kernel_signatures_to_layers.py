#!/usr/bin/env python3
"""Map each unique CUDA kernel signature to its source module/layer in the BLOOM model.

A unique kernel is defined by: name + grid + block + shared_memory

This script processes the PyTorch profiler trace (with layer annotations) and maps
each unique kernel signature to the transformer layer(s) that invoked it.

Usage:
    python map_kernel_signatures_to_layers.py trace_with_layers.json kernel_signatures.json output.json
"""

import json
import sys
import re
from collections import defaultdict
from typing import Dict, List, Set, Tuple


def make_signature_key(name: str, grid: List[int], block: List[int], shared_mem: int) -> str:
    """Create a unique key for a kernel signature."""
    grid_str = f"{grid[0]}x{grid[1]}x{grid[2]}"
    block_str = f"{block[0]}x{block[1]}x{block[2]}"
    return f"{name}|||{grid_str}|||{block_str}|||{shared_mem}"


def main():
    if len(sys.argv) != 4:
        print(f"Usage: {sys.argv[0]} <trace_with_layers.json> <kernel_signatures.json> <output.json>")
        sys.exit(1)
    
    trace_path = sys.argv[1]
    kernel_sigs_path = sys.argv[2]
    output_path = sys.argv[3]
    
    # Load kernel signatures
    print(f"Loading kernel signatures from {kernel_sigs_path}...")
    with open(kernel_sigs_path, 'r') as f:
        kernel_sigs_data = json.load(f)
    
    kernel_sigs = kernel_sigs_data['kernels']
    print(f"Loaded {len(kernel_sigs)} kernel signatures")
    
    # Build signature map
    sig_to_info = {}
    for sig in kernel_sigs:
        name = sig['name']
        tier = sig['tier']
        
        # Tier 1 (memcpy/memset) uses bytes, not grid/block/shared_mem
        if tier == 1:
            bytes_val = sig['signature'].get('bytes', 0)
            # Use a special key for Tier 1
            key = f"{name}|||tier1|||{bytes_val}"
            sig_to_info[key] = {
                'name': name,
                'tier': tier,
                'count': sig['count'],
                'bytes': bytes_val,
                'params': sig.get('params', {})
            }
        else:
            # Tier 2 and 3 use grid/block/shared_mem
            grid = sig['signature'].get('grid', [1, 1, 1])
            block = sig['signature'].get('block', [1, 1, 1])
            shared_mem = sig['signature'].get('shared memory', 0)
            key = make_signature_key(name, grid, block, shared_mem)
            sig_to_info[key] = {
                'name': name,
                'tier': tier,
                'count': sig['count'],
                'grid': grid,
                'block': block,
                'shared_memory': shared_mem,
                'params': sig.get('params', {})
            }
    
    # Load trace
    print(f"\nLoading trace from {trace_path}...")
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
    
    # Step 3: Find cuda_runtime events and their correlation IDs
    print("\nMapping cuda_runtime External IDs to correlation IDs...")
    ext_id_to_corr = {}
    
    for event in events:
        if event.get('cat') == 'cuda_runtime':
            ext_id = event.get('args', {}).get('External id')
            corr = event.get('args', {}).get('correlation')
            if ext_id and corr:
                ext_id_to_corr[ext_id] = corr
    
    print(f"Found {len(ext_id_to_corr)} cuda_runtime events")
    
    # Step 4: Build correlation -> layer mapping
    corr_id_to_layer = {}
    for ext_id, layers in ext_id_to_layer.items():
        if ext_id in ext_id_to_corr:
            corr = ext_id_to_corr[ext_id]
            if corr not in corr_id_to_layer:
                corr_id_to_layer[corr] = set()
            corr_id_to_layer[corr].update(layers)
    
    print(f"Mapped {len(corr_id_to_layer)} correlation IDs to layers")
    
    # Step 5: Map each unique kernel SIGNATURE to layers
    print("\nMapping kernel signatures to layers...")
    sig_to_layers = defaultdict(set)
    sig_to_invocations = defaultdict(int)
    found_sigs = set()
    
    for event in events:
        cat = event.get('cat')
        
        # Handle Kernel events (Tier 2, 3)
        if cat == 'Kernel':
            kernel_name = event.get('name', '')
            args = event.get('args', {})
            grid = args.get('grid', [])
            block = args.get('block', [])
            shared_mem = args.get('shared memory', 0)
            corr = args.get('correlation')
            
            # Create signature key
            sig_key = make_signature_key(kernel_name, grid, block, shared_mem)
            found_sigs.add(sig_key)
            
            if corr in corr_id_to_layer:
                layers = corr_id_to_layer[corr]
                sig_to_layers[sig_key].update(layers)
                sig_to_invocations[sig_key] += 1
        
        # Handle Memcpy/Memset events (Tier 1)
        elif cat in ['Memcpy', 'Memset']:
            kernel_name = event.get('name', '')
            args = event.get('args', {})
            bytes_val = args.get('bytes', 0)
            corr = args.get('correlation')
            
            # Create Tier 1 signature key
            sig_key = f"{kernel_name}|||tier1|||{bytes_val}"
            found_sigs.add(sig_key)
            
            if corr in corr_id_to_layer:
                layers = corr_id_to_layer[corr]
                sig_to_layers[sig_key].update(layers)
                sig_to_invocations[sig_key] += 1
    
    print(f"Found {len(found_sigs)} unique kernel signatures in trace")
    print(f"Mapped {len(sig_to_layers)} kernel signatures to layers")
    
    # Step 6: Match with kernel_signatures.json and build output
    print("\nMatching with kernel_signatures.json...")
    result = []
    matched = 0
    unmatched = 0
    
    for sig_key, sig_info in sig_to_info.items():
        tier = sig_info['tier']
        
        if sig_key in sig_to_layers:
            layers = sorted(list(sig_to_layers[sig_key]))
            invocations_in_trace = sig_to_invocations[sig_key]
            
            # Extract layer numbers
            layer_nums = []
            for layer in layers:
                match = re.search(r'transformer\.h\.(\d+)', layer)
                if match:
                    layer_nums.append(int(match.group(1)))
            
            entry = {
                'kernel_name': sig_info['name'],
                'tier': tier,
                'params': sig_info['params'],
                'invocation_count_expected': sig_info['count'],
                'invocation_count_in_trace': invocations_in_trace,
                'layers': layers,
                'layer_count': len(layers),
                'layer_numbers': sorted(set(layer_nums)),
                'has_layer_attribution': True
            }
            
            # Add tier-specific fields
            if tier == 1:
                entry['bytes'] = sig_info.get('bytes', 0)
            else:
                entry['grid'] = sig_info['grid']
                entry['block'] = sig_info['block']
                entry['shared_memory'] = sig_info['shared_memory']
            
            result.append(entry)
            matched += 1
        else:
            # Kernel signature not found in trace (likely in preprocessing/embedding layers)
            entry = {
                'kernel_name': sig_info['name'],
                'tier': tier,
                'params': sig_info['params'],
                'invocation_count_expected': sig_info['count'],
                'invocation_count_in_trace': 0,
                'layers': [],
                'layer_count': 0,
                'layer_numbers': [],
                'has_layer_attribution': False,
                'note': 'Not found in trace with layer annotations (likely preprocessing/embedding)'
            }
            
            # Add tier-specific fields
            if tier == 1:
                entry['bytes'] = sig_info.get('bytes', 0)
            else:
                entry['grid'] = sig_info['grid']
                entry['block'] = sig_info['block']
                entry['shared_memory'] = sig_info['shared_memory']
            
            result.append(entry)
            unmatched += 1
    
    # Save result
    print(f"\nSaving results to {output_path}...")
    with open(output_path, 'w') as f:
        json.dump({'kernel_signatures': result, 'summary': {
            'total': len(result),
            'matched': matched,
            'unmatched': unmatched
        }}, f, indent=2)
    
    # Print summary statistics
    print("\n=== Summary ===")
    print(f"Total kernel signatures in kernel_signatures.json: {len(sig_to_info)}")
    print(f"Matched with layer attribution: {matched}")
    print(f"Unmatched (no layer attribution): {unmatched}")
    
    if matched > 0:
        kernels_by_layer_count = defaultdict(int)
        for item in result:
            if item['has_layer_attribution']:
                kernels_by_layer_count[item['layer_count']] += 1
        
        print("\nMatched kernels by number of layers:")
        for count in sorted(kernels_by_layer_count.keys()):
            print(f"  {count} layer(s): {kernels_by_layer_count[count]} kernels")
        
        # Show a few examples
        print("\n=== Sample kernel signature mappings ===")
        for item in [x for x in result if x['has_layer_attribution']][:3]:
            print(f"\nKernel: {item['kernel_name'][:80]}")
            if item['tier'] == 1:
                print(f"  Tier: 1, Bytes: {item.get('bytes', 0)}")
            else:
                print(f"  Grid: {item.get('grid')}, Block: {item.get('block')}, Shared mem: {item.get('shared_memory')}")
            print(f"  Appears in {item['layer_count']} layers: {item['layers'][:3]}")
            print(f"  Invocations: expected={item['invocation_count_expected']}, in_trace={item['invocation_count_in_trace']}")
    
    print(f"\nDone! Wrote {len(result)} kernel signature mappings to {output_path}")


if __name__ == '__main__':
    main()
