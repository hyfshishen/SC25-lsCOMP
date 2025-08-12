import os, sys

os.chdir("./lsCOMP")
os.system("mkdir build")
os.chdir("./build")
os.system("cmake ..")
os.system("make -j")
