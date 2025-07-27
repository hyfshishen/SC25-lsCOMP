import os, sys
import subprocess

lsCOMP_path = "./lsCOMP/build/lsCOMP_"
datasets_path = "./datasets/"

# execute cssi
compression_throughput = []
decompression_throughput = []
base_quant = {3, 4, 5, 6, 7, 8, 9, 10}
for quant in base_quant:
    cmd_cssi = f"{lsCOMP_path}cssi -i {datasets_path}cssi-600.bin -d 600 1813 1558 -b {quant} {quant+1} {quant+2} {quant+3} -p 1"
    result = subprocess.run(cmd_cssi, shell=True, capture_output=True, text=True)
    temp_compress_throughput = None
    temp_decompress_throughput = None
    for line in result.stdout.splitlines():
        if "compression   end-to-end speed" in line:
            temp_compress_throughput = float(line.strip().split()[-2])
        elif "decompression end-to-end speed" in line:
            temp_decompress_throughput = float(line.strip().split()[-2])
    compression_throughput.append(temp_compress_throughput)
    decompression_throughput.append(temp_decompress_throughput)
print("(a) CSSI-CMP curve:")
print(compression_throughput)
print("(a) CSSI-DEC curve:")
print(decompression_throughput)

# execute xpcs
compression_throughput = []
decompression_throughput = []
base_quant = {3, 4, 5, 6, 7, 8, 9, 10}
for quant in base_quant:
    cmd_xpcs = f"{lsCOMP_path}xpcs -i {datasets_path}xpcs-512-1.bin -d 512 1813 1558 -b {quant} {quant+1} {quant+2} {quant+3} -p 1"
    result = subprocess.run(cmd_xpcs, shell=True, capture_output=True, text=True)
    temp_compress_throughput = None
    temp_decompress_throughput = None
    for line in result.stdout.splitlines():
        if "compression   end-to-end speed" in line:
            temp_compress_throughput = float(line.strip().split()[-2])
        elif "decompression end-to-end speed" in line:
            temp_decompress_throughput = float(line.strip().split()[-2])
    compression_throughput.append(temp_compress_throughput)
    decompression_throughput.append(temp_decompress_throughput)
print("(a) XPCS-CMP curve:")
print(compression_throughput)
print("(a) XPCS-DEC curve:")
print(decompression_throughput)

# execute cssi
compression_throughput = []
decompression_throughput = []
thresholds = {0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0}
for threshold in thresholds:
    cmd_cssi = f"{lsCOMP_path}cssi -i {datasets_path}cssi-600.bin -d 600 1813 1558 -b 5 5 5 5 -p {threshold}"
    result = subprocess.run(cmd_cssi, shell=True, capture_output=True, text=True)
    temp_compress_throughput = None
    temp_decompress_throughput = None
    for line in result.stdout.splitlines():
        if "compression   end-to-end speed" in line:
            temp_compress_throughput = float(line.strip().split()[-2])
        elif "decompression end-to-end speed" in line:
            temp_decompress_throughput = float(line.strip().split()[-2])
    compression_throughput.append(temp_compress_throughput)
    decompression_throughput.append(temp_decompress_throughput)
print("(b) CSSI-CMP curve:")
print(compression_throughput)
print("(b) CSSI-DEC curve:")
print(decompression_throughput)

# execute xpcs
compression_throughput = []
decompression_throughput = []
thresholds = {0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0}
for threshold in thresholds:
    cmd_xpcs = f"{lsCOMP_path}xpcs -i {datasets_path}xpcs-512-1.bin -d 512 1813 1558 -b 5 5 5 5 -p {threshold}"
    result = subprocess.run(cmd_xpcs, shell=True, capture_output=True, text=True)
    temp_compress_throughput = None
    temp_decompress_throughput = None
    for line in result.stdout.splitlines():
        if "compression   end-to-end speed" in line:
            temp_compress_throughput = float(line.strip().split()[-2])
        elif "decompression end-to-end speed" in line:
            temp_decompress_throughput = float(line.strip().split()[-2])
    compression_throughput.append(temp_compress_throughput)
    decompression_throughput.append(temp_decompress_throughput)
print("(b) CSSI-CMP curve:")
print(compression_throughput)
print("(b) CSSI-DEC curve:")
print(decompression_throughput)