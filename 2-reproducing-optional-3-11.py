import os, sys
import subprocess

lsCOMP_path = "./lsCOMP/build/lsCOMP_xpcs"
datasets_path = "./datasets/"
dataset1 = "chameleon_1024x1024x1080_uint16.raw"
dataset2 = "pawpawsaurus_958x646x1088_uint16.raw"
dataset3 = "spathorhynchus_1024x1024x750_uint16.raw"

# execute dataset1
cmd1 = f"{lsCOMP_path} -i {datasets_path}{dataset1} -d 1080 1024 1024 -b 1 1 1 1 -p 1"
result = subprocess.run(cmd1, shell=True, capture_output=True, text=True)
compression_ratio = None
for line in result.stdout.splitlines():
    if "compression ratio" in line:
        compression_ratio = float(line.strip().split()[-1])
print("lsCOMP for Chameleon, lossless")
print("Compression ratio:", compression_ratio)

# execute dataset2
cmd2 = f"{lsCOMP_path} -i {datasets_path}{dataset2} -d 1088 646 958 -b 1 1 1 1 -p 1"
result = subprocess.run(cmd2, shell=True, capture_output=True, text=True)
compression_ratio = None
for line in result.stdout.splitlines():
    if "compression ratio" in line:
        compression_ratio = float(line.strip().split()[-1])
print("lsCOMP for P. Campbelli, lossless")
print("Compression ratio:", compression_ratio)

# execute dataset3
cmd3 = f"{lsCOMP_path} -i {datasets_path}{dataset3} -d 750 1024 1024 -b 1 1 1 1 -p 1"
result = subprocess.run(cmd3, shell=True, capture_output=True, text=True)
compression_ratio = None
for line in result.stdout.splitlines():
    if "compression ratio" in line:
        compression_ratio = float(line.strip().split()[-1])
print("lsCOMP for S. Fossorium, lossless")
print("Compression ratio:", compression_ratio)