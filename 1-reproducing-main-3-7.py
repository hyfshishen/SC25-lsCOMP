import os, sys
import subprocess

lsCOMP_path = "./lsCOMP/build/lsCOMP_"
datasets_path = "./datasets/"
bandwidths = [1, 2, 4, 8, 16, 32] # in GB/s

# execute cssi
cmd_cssi = f"{lsCOMP_path}cssi -i {datasets_path}cssi-600.bin -d 600 1813 1558 -b 1 1 1 1 -p 1"
for i in range(5):
    result = subprocess.run(cmd_cssi, shell=True, capture_output=True, text=True)
result = subprocess.run(cmd_cssi, shell=True, capture_output=True, text=True)
compress_throughput = None
decompress_throughput = None
compression_ratio = None
for line in result.stdout.splitlines():
    if "compression   end-to-end speed" in line:
        compress_throughput = float(line.strip().split()[-2])
    elif "decompression end-to-end speed" in line:
        decompress_throughput = float(line.strip().split()[-2])
    elif "compression ratio" in line:
        compression_ratio = float(line.strip().split()[-1])
transfer_time = []
for bandwidth in bandwidths:
    overall_time = 100.0 / compress_throughput + 100.0 / compression_ratio / bandwidth
    transfer_time.append(overall_time)
print("Transfering 100 GB CSSI dataset with lsCOMP")
print(transfer_time)

print()

# execute xpcs
cmd_xpcs = f"{lsCOMP_path}xpcs -i {datasets_path}xpcs-512-1.bin -d 512 1813 1558 -b 1 1 1 1 -p 1"
for i in range(5):
    result = subprocess.run(cmd_xpcs, shell=True, capture_output=True, text=True)
result = subprocess.run(cmd_xpcs, shell=True, capture_output=True, text=True)
compress_throughput = None
decompress_throughput = None
compression_ratio = None
for line in result.stdout.splitlines():
    if "compression   end-to-end speed" in line:
        compress_throughput = float(line.strip().split()[-2])
    elif "decompression end-to-end speed" in line:
        decompress_throughput = float(line.strip().split()[-2])
    elif "compression ratio" in line:
        compression_ratio = float(line.strip().split()[-1])
transfer_time = []
for bandwidth in bandwidths:
    overall_time = 100.0 / compress_throughput + 100.0 / compression_ratio / bandwidth
    transfer_time.append(overall_time)
print("Transfering 100 GB XPCS dataset with lsCOMP")
print(transfer_time)