import os
from shutil import copyfile

"""
files = os.listdir("1-100")
c = 2
for i in sorted(files):
    print(f"img_{c:04}.jpg", i)
    copyfile("1-100/" + i, f"img_{c:04}.jpg")
    c += 2

dir_name = "100-300-2"
files = os.listdir(dir_name)
c = 100
for i in sorted(files):
    print(f"img_{c:04}.jpg", i)
    copyfile(dir_name + "/" + i, f"img_{c:04}.jpg")
    c += 4
"""
dir_name = "300-500-2"
files = os.listdir(dir_name)
c = 304
for i in sorted(files):
    print(f"img_{c:04}.jpg", i)
    copyfile(dir_name + "/" + i, f"img_{c:04}.jpg")
    c += 4
