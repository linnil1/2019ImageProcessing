from PIL import Image
import os
import json
import matplotlib.pyplot as plt


dir_name = "images/ori_1-500/"
dir_save = "images/crop_1-500/"
files = os.listdir(dir_name)
for f in sorted(files):
    if not f.endswith(".json"):
        continue
    f = f[:-5]
    print(f)
    img = Image.open(dir_name + f + ".jpg")
    boxes = [i["points"] for i in json.load(open(dir_name + f + ".json"))["shapes"]]
    c = 0
    boxes = [[box[0][0], box[0][1], box[1][0], box[1][1]] for box in boxes]
    for box in sorted(boxes):
        cropimg = img.crop(box).transpose(Image.FLIP_TOP_BOTTOM)
        # img = img.transpose(Image.FLIP_LEFT_RIGHT)
        print(box)
        print(cropimg.size)
        cropimg.save(dir_save + f + f"_{c}.jpg")
        c += 1
        # plt.imshow(cropimg)
        # plt.show()

