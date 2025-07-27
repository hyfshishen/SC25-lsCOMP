import os, sys
import subprocess

lsCOMP_path = "./lsCOMP/build/lsCOMP_"
datasets_path = "./datasets/"

# execute sfc-l
cmd_sfc_l = f"{lsCOMP_path}xpcs -i {datasets_path}0-sfc.uint16 -d 52224 185 194 -b 1 1 1 1 -p 1"
for i in range(5):
    result = subprocess.run(cmd_sfc_l, shell=True, capture_output=True, text=True)
result = subprocess.run(cmd_sfc_l, shell=True, capture_output=True, text=True)
compress_throughput = None
decompress_throughput = None
for line in result.stdout.splitlines():
    if "compression   end-to-end speed" in line:
        compress_throughput = float(line.strip().split()[-2])
    elif "decompression end-to-end speed" in line:
        decompress_throughput = float(line.strip().split()[-2])
print("lsCOMP for SFC-L, lossless")
print("Compression throughput:", compress_throughput, "GB/s")
print("Decompression throughput:", decompress_throughput, "GB/s")

# execute spdi-m
cmd_spdi_m = f"{lsCOMP_path}xpcs -i {datasets_path}1-spdi-m.uint16 -d 841 511 1024 -b 1 1 1 1 -p 1"
for i in range(5):
    result = subprocess.run(cmd_spdi_m, shell=True, capture_output=True, text=True)
result = subprocess.run(cmd_spdi_m, shell=True, capture_output=True, text=True)
compress_throughput = None
decompress_throughput = None
for line in result.stdout.splitlines():
    if "compression   end-to-end speed" in line:
        compress_throughput = float(line.strip().split()[-2])
    elif "decompression end-to-end speed" in line:
        decompress_throughput = float(line.strip().split()[-2])
print("lsCOMP for SPDI-M, lossless")
print("Compression throughput:", compress_throughput, "GB/s")
print("Decompression throughput:", decompress_throughput, "GB/s")

# execute sfc-gi
cmd_sfc_gi = f"{lsCOMP_path}xpcs -i {datasets_path}2-sfc-1.uint16 -d 758 1440 1440 -b 1 1 1 1 -p 1"
for i in range(5):
    result = subprocess.run(cmd_sfc_gi, shell=True, capture_output=True, text=True)
result = subprocess.run(cmd_sfc_gi, shell=True, capture_output=True, text=True)
compress_throughput = None
decompress_throughput = None
for line in result.stdout.splitlines():
    if "compression   end-to-end speed" in line:
        compress_throughput = float(line.strip().split()[-2])
    elif "decompression end-to-end speed" in line:
        decompress_throughput = float(line.strip().split()[-2])
print("lsCOMP for SFC-GI, lossless")
print("Compression throughput:", compress_throughput, "GB/s")
print("Decompression throughput:", decompress_throughput, "GB/s")

# execute pcg-gb
cmd_pcg_gb = f"{lsCOMP_path}xpcs -i {datasets_path}3-pcg.uint16 -d 800 621 621 -b 1 1 1 1 -p 1"
for i in range(5):
    result = subprocess.run(cmd_pcg_gb, shell=True, capture_output=True, text=True)
result = subprocess.run(cmd_pcg_gb, shell=True, capture_output=True, text=True)
compress_throughput = None
decompress_throughput = None
for line in result.stdout.splitlines():
    if "compression   end-to-end speed" in line:
        compress_throughput = float(line.strip().split()[-2])
    elif "decompression end-to-end speed" in line:
        decompress_throughput = float(line.strip().split()[-2])
print("lsCOMP for PCG-GB, lossless")
print("Compression throughput:", compress_throughput, "GB/s")
print("Decompression throughput:", decompress_throughput, "GB/s")