import os, sys


url1 = "http://klacansky.com/open-scivis-datasets/chameleon/chameleon_1024x1024x1080_uint16.raw"
url2 = "http://klacansky.com/open-scivis-datasets/pawpawsaurus/pawpawsaurus_958x646x1088_uint16.raw"
url3 = "http://klacansky.com/open-scivis-datasets/spathorhynchus/spathorhynchus_1024x1024x750_uint16.raw"

os.system(f"wget {url1}")
os.system(f"wget {url2}")
os.system(f"wget {url3}")
os.system("mv *_uint16.raw datasets/")