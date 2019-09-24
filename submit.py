import os
import argparse
import shutil

# read arg
parser = argparse.ArgumentParser(description="Wrap homework to what Teacher want")
parser.add_argument('hw', type=int)
parser.add_argument('id', type=str)
args = parser.parse_args()

# ignore item
ign = open(".gitignore").readlines()
ign = [i.replace("\n", "") for i in ign]
tree_ign = shutil.ignore_patterns(*ign)

# copy dictionary
name = f"HW{args.hw}_{args.id}"
base = f"Homework{args.hw}"
src  = f"{base}/{name} Program folder"
shutil.rmtree(base, ignore_errors=True)
shutil.copytree(f"hw{args.hw}", src, ignore=tree_ign)

# change name
shutil.move(f"{src}/hw{args.hw}.pdf", f"{base}/{name}.pdf")
shutil.copy2(f"{src}/run.sh", f"{src}/HW{args.hw}.sh")

print("Finish: ")
print(os.system(f"tree {base}"))
