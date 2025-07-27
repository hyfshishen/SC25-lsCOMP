import os, sys
import subprocess

lsCOMP_path = "./lsCOMP/build/lsCOMP_"
datasets_path = "./datasets/"

# execute cssi
# base quant 3, pooling threshold 0.7
cmd_cssi = f"{lsCOMP_path}cssi -i {datasets_path}cssi-600.bin -d 600 1813 1558 -b 3 4 5 6 -p 0.7"
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
print("lsCOMP for CSSI, lossy, error=3")
print("Compression throughput:", compress_throughput, "GB/s")
print("Decompression throughput:", decompress_throughput, "GB/s")

# base quant 5, pooling threshold 0.7
cmd_cssi = f"{lsCOMP_path}cssi -i {datasets_path}cssi-600.bin -d 600 1813 1558 -b 5 6 7 8 -p 0.7"
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
print("lsCOMP for CSSI, lossy, error=5")
print("Compression throughput:", compress_throughput, "GB/s")
print("Decompression throughput:", decompress_throughput, "GB/s")

# base quant 10, pooling threshold 0.7
cmd_cssi = f"{lsCOMP_path}cssi -i {datasets_path}cssi-600.bin -d 600 1813 1558 -b 10 11 12 13 -p 0.7"
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
print("lsCOMP for CSSI, lossy, error=10")
print("Compression throughput:", compress_throughput, "GB/s")
print("Decompression throughput:", decompress_throughput, "GB/s")

print()

# execute xpcs
# base quant 3, pooling threshold 0.7
cmd_xpcs = f"{lsCOMP_path}xpcs -i {datasets_path}xpcs-512-1.bin -d 512 1813 1558 -b 3 4 5 6 -p 0.7"
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
print("lsCOMP for XPCS, lossy, error=3")
print("Compression throughput:", compress_throughput, "GB/s")
print("Decompression throughput:", decompress_throughput, "GB/s")

# base quant 5, pooling threshold 0.7
cmd_xpcs = f"{lsCOMP_path}xpcs -i {datasets_path}xpcs-512-1.bin -d 512 1813 1558 -b 5 6 7 8 -p 0.7"
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
print("lsCOMP for XPCS, lossy, error=5")
print("Compression throughput:", compress_throughput, "GB/s")
print("Decompression throughput:", decompress_throughput, "GB/s")

# base quant 10, pooling threshold 0.7
cmd_xpcs = f"{lsCOMP_path}xpcs -i {datasets_path}xpcs-512-1.bin -d 512 1813 1558 -b 10 11 12 13 -p 0.7"
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
print("lsCOMP for XPCS, lossy, error=10")
print("Compression throughput:", compress_throughput, "GB/s")
print("Decompression throughput:", decompress_throughput, "GB/s")