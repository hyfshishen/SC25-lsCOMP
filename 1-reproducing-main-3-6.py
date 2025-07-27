import os, sys
import subprocess

lsCOMP_path = "./lsCOMP/build/lsCOMP_"
datasets_path = "./datasets/"

# execute cssi
compression_ratios = []
base_quant = {3, 4, 5, 6, 7, 8, 9, 10}
for quant in base_quant:
    cmd_cssi = f"{lsCOMP_path}cssi -i {datasets_path}cssi-600.bin -d 600 1813 1558 -b {quant} {quant+1} {quant+2} {quant+3} -p 1"
    result = subprocess.run(cmd_cssi, shell=True, capture_output=True, text=True)
    temp_compression_ratios = None
    for line in result.stdout.splitlines():
        if "compression ratio: " in line:
            temp_compression_ratios = float(line.strip().split()[-1])
    compression_ratios.append(temp_compression_ratios)
print("(a) CSSI-CR curve:")
print(compression_ratios)

# execute xpcs
compression_ratios = []
base_quant = {3, 4, 5, 6, 7, 8, 9, 10}
for quant in base_quant:
    cmd_xpcs = f"{lsCOMP_path}xpcs -i {datasets_path}xpcs-512-1.bin -d 512 1813 1558 -b {quant} {quant+1} {quant+2} {quant+3} -p 1"
    result = subprocess.run(cmd_xpcs, shell=True, capture_output=True, text=True)
    temp_compression_ratios = None
    for line in result.stdout.splitlines():
        if "compression ratio: " in line:
            temp_compression_ratios = float(line.strip().split()[-1])
    compression_ratios.append(temp_compression_ratios)
print("(a) XPCS-CR curve:")
print(compression_ratios)

# execute cssi
compression_ratios = []
thresholds = {0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0}
for threshold in thresholds:
    cmd_cssi = f"{lsCOMP_path}cssi -i {datasets_path}cssi-600.bin -d 600 1813 1558 -b 3 3 3 3 -p {threshold}"
    result = subprocess.run(cmd_cssi, shell=True, capture_output=True, text=True)
    temp_compression_ratios = None
    for line in result.stdout.splitlines():
        if "compression ratio: " in line:
            temp_compression_ratios = float(line.strip().split()[-1])
    compression_ratios.append(temp_compression_ratios)
print("(b) CSSI-CR curve:")
print(compression_ratios)

# execute xpcs
compression_ratios = []
thresholds = {0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0}
for threshold in thresholds:
    cmd_xpcs = f"{lsCOMP_path}xpcs -i {datasets_path}xpcs-512-1.bin -d 512 1813 1558 -b 3 3 3 3 -p {threshold}"
    result = subprocess.run(cmd_xpcs, shell=True, capture_output=True, text=True)
    temp_compression_ratios = None
    for line in result.stdout.splitlines():
        if "compression ratio: " in line:
            temp_compression_ratios = float(line.strip().split()[-1])
    compression_ratios.append(temp_compression_ratios)
print("(b) XPCS-CR curve:")
print(compression_ratios)