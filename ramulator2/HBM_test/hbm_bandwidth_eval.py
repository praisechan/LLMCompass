#!/usr/bin/env python3
"""
HBM Bandwidth Evaluation Script for Ramulator2

This script:
1. Generates memory traces using tracegen.py
2. Runs Ramulator2 simulation with HBM3 configuration
3. Parses HBM3 specifications directly from HBM3.cpp source code
4. Calculates effective and theoretical bandwidth
5. Reports bandwidth efficiency

Fixed issues:
- Now reads HBM3 configuration from HBM3.cpp instead of YAML config
- Correctly handles organization and timing presets
- Properly calculates theoretical bandwidth using HBM3 specifications
- Supports channel overrides from YAML config
"""

import os
import subprocess
import yaml
import re

# --- CONFIGURABLE ---
NUM_INSTS = "100"  # Number of instructions to generate
HBM_TEST_DIR = "HBM_test"
TRACEGEN = f"./{HBM_TEST_DIR}/tracegen.py"
TRACE_NAME = f"./{HBM_TEST_DIR}/trace/test_LStrace_{NUM_INSTS}.txt"
CONFIG_NAME = f"./{HBM_TEST_DIR}/hbm_config.yaml"
RAMULATOR = "./ramulator2"
OUTPUT_NAME = "debug.txt"
HBM3_CPP_FILE = "./src/dram/impl/HBM3.cpp"
# Function to parse HBM3 specifications from C++ source
def parse_hbm3_specs(hbm3_file):
    """Parse HBM3 organization and timing presets from C++ source file"""
    with open(hbm3_file, 'r') as f:
        content = f.read()
    
    # Parse organization presets
    org_match = re.search(r'inline static const std::map<std::string, Organization> org_presets = \{(.*?)\};', content, re.DOTALL)
    if not org_match:
        raise RuntimeError("Could not find organization presets in HBM3.cpp")
    
    org_presets = {}
    org_lines = org_match.group(1).strip()
    for line in org_lines.split('\n'):
        if line.strip().startswith('{"') and not line.strip().startswith('//'):
            # Parse line like: {"HBM3_8Gb", {8 << 10, 128, {1, 2, 4, 4, 1 << 15, 1 << 6}}},
            match = re.search(r'\{"([^"]+)",\s*\{([^}]+),\s*(\d+),\s*\{([^}]+)\}\}\}', line)
            if match:
                name = match.group(1)
                density = eval(match.group(2))  # Evaluate expressions like "8 << 10"
                dq = int(match.group(3))
                counts = [eval(x.strip()) for x in match.group(4).split(',')]
                org_presets[name] = {
                    'density': density,
                    'dq': dq,
                    'counts': counts  # [channel, pseudochannel, bankgroup, bank, row, column]
                }
    
    # Parse timing presets
    timing_match = re.search(r'inline static const std::map<std::string, std::vector<int>> timing_presets = \{(.*?)\};', content, re.DOTALL)
    if not timing_match:
        raise RuntimeError("Could not find timing presets in HBM3.cpp")
    
    timing_presets = {}
    timing_lines = timing_match.group(1).strip()
    for line in timing_lines.split('\n'):
        if line.strip().startswith('{"') and not line.strip().startswith('//'):
            # Parse line like: {"HBM3_5.2Gbps", {5200, 4, 7, ...}},
            match = re.search(r'\{"([^"]+)",\s*\{([^}]+)\}\}', line)
            if match:
                name = match.group(1)
                values = [int(x.strip()) for x in match.group(2).split(',')]
                timing_presets[name] = {
                    'rate': values[0],  # MT/s
                    'tCK_ps': values[-1] if values[-1] != -1 else None
                }
    
    return org_presets, timing_presets

def get_hbm_config_from_yaml(config_file):
    """Extract HBM configuration from YAML config file"""
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
    
    # Find the memory device configuration
    memory_system = config.get("MemorySystem", {})
    device = memory_system.get("DRAM", [])
    
    # if not devices:
    #     raise RuntimeError("No memory devices found in config")
    
    # device = devices[0]
    
    # Extract organization and timing preset names
    org_preset = device.get("org", {}).get("preset", None)
    timing_preset = device.get("timing", {}).get("preset", None)
    
    # Also try to get any overrides
    channel_override = device.get("org", {}).get("channel", None)
    
    return org_preset, timing_preset, channel_override

# 1. Generate trace
if os.path.exists(TRACE_NAME):
    print("[INFO] Trace already exists.")
else:
    print("[INFO] Generating trace with tracegen.py...")
    subprocess.run([
        "python", TRACEGEN, "--type", "LStrace", "--pattern", "stream", "--num-insts", NUM_INSTS, "--output", TRACE_NAME
    ], check=True)

# 2. Copy YAML config if needed
if not os.path.exists(CONFIG_NAME):
    for f in os.listdir(HBM_TEST_DIR):
        if f.endswith(".yaml"):
            print(f"[INFO] Copying config {f} from {HBM_TEST_DIR}")
            os.system(f"cp {os.path.join(HBM_TEST_DIR, f)} {CONFIG_NAME}")
            break

# 3. Run Ramulator2
print("[INFO] Running Ramulator2...")
with open(OUTPUT_NAME, "w") as out:
    subprocess.run([
        RAMULATOR, "-f", CONFIG_NAME, "--param", f"trace_file={TRACE_NAME}"
    ], stdout=out, check=True)

# 4. Parse output for memory_system_cycles
with open(OUTPUT_NAME) as f:
    txt = f.read()
match = re.search(r"memory_system_cycles:\s*(\d+)", txt)
if not match:
    raise RuntimeError("memory_system_cycles not found in output!")
sim_cycles = int(match.group(1))
print(f"[INFO] Simulation cycles: {sim_cycles}")

# 5. Parse trace for total bytes transferred (assume each line is a transaction, 64B per transaction)
with open(TRACE_NAME) as f:
    num_txns = sum(1 for _ in f)
if num_txns != int(NUM_INSTS):
    raise ValueError("NUM_INSTS should be the same as num_txns")
BYTES_PER_TXN = 64
bytes_total = num_txns * BYTES_PER_TXN
print(f"[INFO] Total transactions: {num_txns}, total bytes: {bytes_total}")

# 6. Parse HBM3 specifications from C++ source
if not os.path.exists(HBM3_CPP_FILE):
    raise RuntimeError(f"HBM3.cpp file not found at {HBM3_CPP_FILE}")

print("[INFO] Parsing HBM3 specifications from C++ source...")
org_presets, timing_presets = parse_hbm3_specs(HBM3_CPP_FILE)

print(f"[INFO] Found {len(org_presets)} organization presets: {list(org_presets.keys())}")
print(f"[INFO] Found {len(timing_presets)} timing presets: {list(timing_presets.keys())}")

# 7. Get HBM configuration from YAML config
print("[INFO] Reading HBM configuration from YAML...")
org_preset_name, timing_preset_name, channel_override = get_hbm_config_from_yaml(CONFIG_NAME)

if not org_preset_name or org_preset_name not in org_presets:
    raise RuntimeError(f"Organization preset '{org_preset_name}' not found in HBM3.cpp")

if not timing_preset_name or timing_preset_name not in timing_presets:
    raise RuntimeError(f"Timing preset '{timing_preset_name}' not found in HBM3.cpp")

org_spec = org_presets[org_preset_name]
timing_spec = timing_presets[timing_preset_name]

print(f"[INFO] Using organization preset: {org_preset_name}")
print(f"[INFO] Using timing preset: {timing_preset_name}")

# Extract parameters
channels = channel_override if channel_override is not None else org_spec['counts'][0]  # channel count
dq_width = org_spec['dq']         # data width per channel (bits)
rate_mts = timing_spec['rate']    # Transfer rate in MT/s
tCK_ps = timing_spec['tCK_ps']    # Clock period in picoseconds

if tCK_ps is None:
    # Calculate tCK from rate: rate = 2 * freq, so freq = rate/2, tCK = 1/freq
    freq_hz = (rate_mts / 2) * 1e6  # Convert MT/s to Hz
    tCK_ps = 1e12 / freq_hz  # Convert to picoseconds

print(f"[INFO] HBM3 Configuration:")
print(f"  Channels: {channels}")
print(f"  Data width per channel: {dq_width} bits")
print(f"  Transfer rate: {rate_mts} MT/s")
print(f"  Clock period: {tCK_ps:.2f} ps")

# 8. Calculate effective bandwidth
# 8. Calculate effective bandwidth
sim_time_ns = sim_cycles * tCK_ps  # Convert to nanoseconds (ps * 1e-3 = ns)
sim_time_s = sim_time_ns * 1e-12   # Convert to seconds (ps * 1e-12 = s)
bw_eff = bytes_total / sim_time_s / 1e9  # GB/s
print(f"[RESULT] Effective Bandwidth: {bw_eff:.2f} GB/s")

# 9. Calculate theoretical bandwidth
# Theoretical BW = channels * data_width(bits) * transfer_rate(MT/s) / 8(bits_to_bytes) / 1000(MB_to_GB)
# Note: MT/s means Million Transfers per second, each transfer moves data_width bits
bw_theory = channels * dq_width * rate_mts / 8 / 1000  # GB/s
print(f"[RESULT] Theoretical Bandwidth: {bw_theory:.2f} GB/s")
print(f"[RESULT] Efficiency: {bw_eff / bw_theory * 100:.2f}%")
