import os, sys
import subprocess

lsCOMP_path = "./lsCOMP/build/lsCOMP_"
datasets_path = "./datasets/"

# execute cssi
# error=3
compression_ratios = []
cmd_cssi = f"{lsCOMP_path}cssi -i {datasets_path}cssi-600.bin -d 600 1813 1558 -b 3 4 5 6 -p 0.7"
result = subprocess.run(cmd_cssi, shell=True, capture_output=True, text=True)
for line in result.stdout.splitlines():
    if "compression ratio: " in line:
        compression_ratios.append(float(line.strip().split()[-1]))
cmd_cssi = f"{lsCOMP_path}cssi -i {datasets_path}cssi-128.bin -d 128 1813 1558 -b 3 4 5 6 -p 0.7"
result = subprocess.run(cmd_cssi, shell=True, capture_output=True, text=True)
for line in result.stdout.splitlines():
    if "compression ratio: " in line:
        compression_ratios.append(float(line.strip().split()[-1]))
print("lsCOMP for CSSI, lossy, error=3")
print(f"Compression ratios: {min(compression_ratios)} -- {max(compression_ratios)}")

# error=5
compression_ratios = []
cmd_cssi = f"{lsCOMP_path}cssi -i {datasets_path}cssi-600.bin -d 600 1813 1558 -b 5 6 7 8 -p 0.7"
result = subprocess.run(cmd_cssi, shell=True, capture_output=True, text=True)
for line in result.stdout.splitlines():
    if "compression ratio: " in line:
        compression_ratios.append(float(line.strip().split()[-1]))
cmd_cssi = f"{lsCOMP_path}cssi -i {datasets_path}cssi-128.bin -d 128 1813 1558 -b 5 6 7 8 -p 0.7"
result = subprocess.run(cmd_cssi, shell=True, capture_output=True, text=True)
for line in result.stdout.splitlines():
    if "compression ratio: " in line:
        compression_ratios.append(float(line.strip().split()[-1]))
print("lsCOMP for CSSI, lossy, error=5")
print(f"Compression ratios: {min(compression_ratios)} -- {max(compression_ratios)}")

# error=10
compression_ratios = []
cmd_cssi = f"{lsCOMP_path}cssi -i {datasets_path}cssi-600.bin -d 600 1813 1558 -b 10 11 12 13 -p 0.7"
result = subprocess.run(cmd_cssi, shell=True, capture_output=True, text=True)
for line in result.stdout.splitlines():
    if "compression ratio: " in line:
        compression_ratios.append(float(line.strip().split()[-1]))
cmd_cssi = f"{lsCOMP_path}cssi -i {datasets_path}cssi-128.bin -d 128 1813 1558 -b 10 11 12 13 -p 0.7"
result = subprocess.run(cmd_cssi, shell=True, capture_output=True, text=True)
for line in result.stdout.splitlines():
    if "compression ratio: " in line:
        compression_ratios.append(float(line.strip().split()[-1]))
print("lsCOMP for CSSI, lossy, error=10")
print(f"Compression ratios: {min(compression_ratios)} -- {max(compression_ratios)}")

print()

# execute xpcs
# error=3
compression_ratios = []
snapshots = ["xpcs-512-1.bin", "xpcs-512-2.bin", "xpcs-512-3.bin", "xpcs-512-4.bin", "xpcs-512-5.bin", "xpcs-512-6.bin"]
for snapshot in snapshots:
    cmd_xpcs = f"{lsCOMP_path}xpcs -i {datasets_path}{snapshot} -d 512 1813 1558 -b 3 4 5 6 -p 0.7"
    result = subprocess.run(cmd_xpcs, shell=True, capture_output=True, text=True)
    for line in result.stdout.splitlines():
        if "compression ratio: " in line:
            compression_ratios.append(float(line.strip().split()[-1]))
print("lsCOMP for XPCS, lossy, error=3")
print(f"Compression ratios: {min(compression_ratios)} -- {max(compression_ratios)}")

# error=5
compression_ratios = []
snapshots = ["xpcs-512-1.bin", "xpcs-512-2.bin", "xpcs-512-3.bin", "xpcs-512-4.bin", "xpcs-512-5.bin", "xpcs-512-6.bin"]
for snapshot in snapshots:
    cmd_xpcs = f"{lsCOMP_path}xpcs -i {datasets_path}{snapshot} -d 512 1813 1558 -b 5 6 7 8 -p 0.7"
    result = subprocess.run(cmd_xpcs, shell=True, capture_output=True, text=True)
    for line in result.stdout.splitlines():
        if "compression ratio: " in line:
            compression_ratios.append(float(line.strip().split()[-1]))
print("lsCOMP for XPCS, lossy, error=5")
print(f"Compression ratios: {min(compression_ratios)} -- {max(compression_ratios)}")

# error=10
compression_ratios = []
snapshots = ["xpcs-512-1.bin", "xpcs-512-2.bin", "xpcs-512-3.bin", "xpcs-512-4.bin", "xpcs-512-5.bin", "xpcs-512-6.bin"]
for snapshot in snapshots:
    cmd_xpcs = f"{lsCOMP_path}xpcs -i {datasets_path}{snapshot} -d 512 1813 1558 -b 10 11 12 13 -p 0.7"
    result = subprocess.run(cmd_xpcs, shell=True, capture_output=True, text=True)
    for line in result.stdout.splitlines():
        if "compression ratio: " in line:
            compression_ratios.append(float(line.strip().split()[-1]))
print("lsCOMP for XPCS, lossy, error=10")
print(f"Compression ratios: {min(compression_ratios)} -- {max(compression_ratios)}")