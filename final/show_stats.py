import os
import numpy as np
import matplotlib.pyplot as plt
import json
import argparse


parser = argparse.ArgumentParser(description='PyTorch implementation of DiscoGAN')
parser.add_argument('--task_name', type=str, default='draw', help='Set data name')
parser.add_argument('--model_path', type=str, default='./models/', help='Set the path for trained models')
parser.add_argument('--model_arch', type=str, default='discogan', help='choose among gan/recongan/discogan. gan - standard GAN, recongan - GAN with reconstruction, discogan - DiscoGAN.')
parser.add_argument('--from_index', type=int, default=0, help='From which data')
args = parser.parse_args()

model_path = os.path.join( args.model_path, args.task_name )
model_path = os.path.join( model_path, args.model_arch )
stats_path = os.path.join( model_path, "stats.json" )

stats = json.load(open(stats_path))["data"]

iters = [a["iter"] for a in stats]
gan_loss_A   = [a["genlossA"]   for a in stats]
gan_loss_B   = [a["genlossB"]   for a in stats]
recon_loss_A = [a["reconlossA"] for a in stats]
recon_loss_B = [a["reconlossB"] for a in stats]
dis_loss_A   = [a["dislossA"]   for a in stats]
dis_loss_B   = [a["dislossB"]   for a in stats]

f = args.from_index
iters = iters[f:]

plt.figure(figsize=(8, 10))
plt.subplot(311)
plt.plot(iters, gan_loss_A[f:], label="ganA")
plt.plot(iters, gan_loss_B[f:], label="ganB")
plt.legend()
plt.subplot(312)
plt.plot(iters, recon_loss_A[f:], label="reconA")
plt.plot(iters, recon_loss_B[f:], label="reconB")
plt.legend()
plt.subplot(313)
plt.plot(iters, dis_loss_A[f:], label="disA")
plt.plot(iters, dis_loss_B[f:], label="disB")
plt.legend()
plt.show()
