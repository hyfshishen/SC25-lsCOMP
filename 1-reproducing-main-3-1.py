import os, sys
import subprocess

lsCOMP_path = "./lsCOMP/build/lsCOMP_"
datasets_path = "./datasets/"

# execute cssi
cmd_cssi = f"{lsCOMP_path}cssi -i {datasets_path}cssi-600.bin -d 600 1813 1558 -b 1 1 1 1 -p 1"
for i in range(5):
    result = subprocess.run(cmd_cssi, shell=True, capture_output=True, text=True)
result = subprocess.run(cmd_cssi, shell=True, capture_output=True, text=True)
compress_throughput = None
decompress_throughput = None
for line in result.stdout.splitlines():
    if "compression   end-to-end speed" in line:
        compress_throughput = float(line.strip().split()[-2])
    elif "decompression end-to-end speed" in line:
        decompress_throughput = float(line.strip().split()[-2])
print("lsCOMP for CSSI, lossless")
print("Compression throughput:", compress_throughput, "GB/s")
print("Decompression throughput:", decompress_throughput, "GB/s")

print()

# execute xpcs
cmd_xpcs = f"{lsCOMP_path}xpcs -i {datasets_path}xpcs-512-1.bin -d 512 1813 1558 -b 1 1 1 1 -p 1"
for i in range(5):
    result = subprocess.run(cmd_xpcs, shell=True, capture_output=True, text=True)
result = subprocess.run(cmd_xpcs, shell=True, capture_output=True, text=True)
compress_throughput = None
decompress_throughput = None
for line in result.stdout.splitlines():
    if "compression   end-to-end speed" in line:
        compress_throughput = float(line.strip().split()[-2])
    elif "decompression end-to-end speed" in line:
        decompress_throughput = float(line.strip().split()[-2])
print("lsCOMP for XPCS, lossless")
print("Compression throughput:", compress_throughput, "GB/s")
print("Decompression throughput:", decompress_throughput, "GB/s")