import os
import argparse
import matplotlib.pyplot as plt
from PIL import Image
from progressbar import ETA, Bar, Percentage, ProgressBar
import json


parser = argparse.ArgumentParser(description='PyTorch implementation of DiscoGAN')
parser.add_argument('--task_name', type=str, default='draw', help='Set data name')
parser.add_argument('--result_path', type=str, default='./results/', help='Set the path the result images will be saved.')
parser.add_argument('--model_arch', type=str, default='discogan', help='choose among gan/recongan/discogan. gan - standard GAN, recongan - GAN with reconstruction, discogan - DiscoGAN.')
parser.add_argument('--want', type=int, default=0, help="Show n'th test image")
parser.add_argument('--epoch', type=int, default=-1, help="Show n'th epoch n'th testimage")

args = parser.parse_args()
result_path = os.path.join(args.result_path, args.task_name )
result_path = os.path.join(result_path, args.model_arch )

epoches = [name for name in os.listdir(result_path) if os.path.isdir(os.path.join(result_path, name))]
epoches = sorted(zip(map(float, epoches), epoches))
last_epoch = list(epoches)[args.epoch][-1]
last_epoch_path = os.path.join(result_path, last_epoch)
print(last_epoch)

res = sorted(os.listdir(last_epoch_path))
want = args.want
want_res = res[want * 6: want * 6 + 6]
plt.figure(figsize=(12, 8))
for i in range(6):
    plt.subplot(231 + i)
    plt.title(want_res[i])
    plt.imshow(Image.open(os.path.join(last_epoch_path, want_res[i])))
plt.show()
