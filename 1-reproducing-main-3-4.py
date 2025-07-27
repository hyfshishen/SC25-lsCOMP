import os, sys
import subprocess

lsCOMP_path = "./lsCOMP/build/lsCOMP_"
datasets_path = "./datasets/"

# execute cssi
compression_ratios = []
cmd_cssi = f"{lsCOMP_path}cssi -i {datasets_path}cssi-600.bin -d 600 1813 1558 -b 1 1 1 1 -p 1"
result = subprocess.run(cmd_cssi, shell=True, capture_output=True, text=True)
for line in result.stdout.splitlines():
    if "compression ratio: " in line:
        compression_ratios.append(float(line.strip().split()[-1]))
cmd_cssi = f"{lsCOMP_path}cssi -i {datasets_path}cssi-128.bin -d 128 1813 1558 -b 1 1 1 1 -p 1"
result = subprocess.run(cmd_cssi, shell=True, capture_output=True, text=True)
for line in result.stdout.splitlines():
    if "compression ratio: " in line:
        compression_ratios.append(float(line.strip().split()[-1]))
print("lsCOMP for CSSI, lossless")
print(f"Compression ratios: {min(compression_ratios)} -- {max(compression_ratios)}")

print()

# execute xpcs
compression_ratios = []
snapshots = ["xpcs-512-1.bin", "xpcs-512-2.bin", "xpcs-512-3.bin", "xpcs-512-4.bin", "xpcs-512-5.bin", "xpcs-512-6.bin"]
for snapshot in snapshots:
    cmd_xpcs = f"{lsCOMP_path}xpcs -i {datasets_path}{snapshot} -d 512 1813 1558 -b 1 1 1 1 -p 1"
    result = subprocess.run(cmd_xpcs, shell=True, capture_output=True, text=True)
    for line in result.stdout.splitlines():
        if "compression ratio: " in line:
            compression_ratios.append(float(line.strip().split()[-1]))
print("lsCOMP for XPCS, lossless")
print(f"Compression ratios: {min(compression_ratios)} -- {max(compression_ratios)}")