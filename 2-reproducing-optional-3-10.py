import os, sys
import subprocess

lsCOMP_path = "./lsCOMP/build/lsCOMP_xpcs"
datasets_path = "./datasets/"
dataset1 = "chameleon_1024x1024x1080_uint16.raw"
dataset2 = "pawpawsaurus_958x646x1088_uint16.raw"
dataset3 = "spathorhynchus_1024x1024x750_uint16.raw"

# execute dataset1
cmd1 = f"{lsCOMP_path} -i {datasets_path}{dataset1} -d 1080 1024 1024 -b 1 1 1 1 -p 1"
for i in range(5):
    result = subprocess.run(cmd1, shell=True, capture_output=True, text=True)
result = subprocess.run(cmd1, shell=True, capture_output=True, text=True)
compress_throughput = None
decompress_throughput = None
for line in result.stdout.splitlines():
    if "compression   end-to-end speed" in line:
        compress_throughput = float(line.strip().split()[-2])
    elif "decompression end-to-end speed" in line:
        decompress_throughput = float(line.strip().split()[-2])
print("lsCOMP for Chameleon, lossless")
print("Compression throughput:", compress_throughput, "GB/s")
print("Decompression throughput:", decompress_throughput, "GB/s")

# execute dataset2
cmd2 = f"{lsCOMP_path} -i {datasets_path}{dataset2} -d 1088 646 958 -b 1 1 1 1 -p 1"
for i in range(5):
    result = subprocess.run(cmd2, shell=True, capture_output=True, text=True)
result = subprocess.run(cmd2, shell=True, capture_output=True, text=True)
compress_throughput = None
decompress_throughput = None
for line in result.stdout.splitlines():
    if "compression   end-to-end speed" in line:
        compress_throughput = float(line.strip().split()[-2])
    elif "decompression end-to-end speed" in line:
        decompress_throughput = float(line.strip().split()[-2])
print("lsCOMP for P. Campbelli, lossless")
print("Compression throughput:", compress_throughput, "GB/s")
print("Decompression throughput:", decompress_throughput, "GB/s")

# execute dataset3
cmd3 = f"{lsCOMP_path} -i {datasets_path}{dataset3} -d 750 1024 1024 -b 1 1 1 1 -p 1"
for i in range(5):
    result = subprocess.run(cmd3, shell=True, capture_output=True, text=True)
result = subprocess.run(cmd3, shell=True, capture_output=True, text=True)
compress_throughput = None
decompress_throughput = None
for line in result.stdout.splitlines():
    if "compression   end-to-end speed" in line:
        compress_throughput = float(line.strip().split()[-2])
    elif "decompression end-to-end speed" in line:
        decompress_throughput = float(line.strip().split()[-2])
print("lsCOMP for S. Fossorium, lossless")
print("Compression throughput:", compress_throughput, "GB/s")
print("Decompression throughput:", decompress_throughput, "GB/s")