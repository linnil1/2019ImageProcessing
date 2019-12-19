import os
import argparse
from itertools import chain
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from dataset import *
from model import *
from PIL import Image
from progressbar import ETA, Bar, Percentage, ProgressBar
import json
import pytorch_ssim


parser = argparse.ArgumentParser(description='PyTorch implementation of DiscoGAN')
parser.add_argument('--cuda', type=str, default='true', help='Set cuda usage')
parser.add_argument('--task_name', type=str, default='draw', help='Set data name')
parser.add_argument('--epoch_size', type=int, default=5000, help='Set epoch size')
parser.add_argument('--batch_size', type=int, default=32, help='Set batch size')
parser.add_argument('--no_invert', action='store_false', default=True, help='Invert white and black')
parser.add_argument('--no_aspect', action='store_false', default=True, help='Maintaain aspect ratio')
parser.add_argument('--result_path', type=str, default='./results/', help='Set the path the result images will be saved.')
parser.add_argument('--model_path', type=str, default='./models/', help='Set the path for trained models')
parser.add_argument('--model_arch', type=str, default='discogan', help='choose among gan/recongan/discogan. gan - standard GAN, recongan - GAN with reconstruction, discogan - DiscoGAN.')
parser.add_argument('--image_size', type=int, default=256, help='Image size. 64 for every experiment in the paper')

parser.add_argument('--learning_rate', type=float, default=0.001, help='Set learning rate for optimizer')
parser.add_argument('--gan_curriculum', type=int, default=10000, help='Strong GAN loss for certain period at the beginning')
parser.add_argument('--starting_rate', type=float, default=0.01, help='Set the lambda weight between GAN loss and Recon loss during curriculum period at the beginning. We used the 0.01 weight.')
parser.add_argument('--default_rate', type=float, default=0.5, help='Set the lambda weight between GAN loss and Recon loss after curriculum period. We used the 0.5 weight.')

parser.add_argument('--n_test', type=int, default=50, help='Number of test data.')

parser.add_argument('--update_interval', type=int, default=5, help='')
parser.add_argument('--log_interval', type=int, default=50, help='Print loss values every log_interval iterations.')
parser.add_argument('--image_save_interval', type=int, default=1000, help='Save test results every image_save_interval iterations.')
parser.add_argument('--model_save_interval', type=int, default=10000, help='Save models every model_save_interval iterations.')


def as_np(data):
    return data.cpu().data.numpy()


def get_data():
    if args.task_name == 'draw':
        data_A, data_B = get_draw_files(test=False, n_test=args.n_test)
        test_A, test_B = get_draw_files(test=True,  n_test=args.n_test)

    else:
        raise ValueError("Not draw dataset")

    return data_A, data_B, test_A, test_B


def get_gan_loss(dis_real, dis_fake, criterion, cuda):
    labels_dis_real = Variable(torch.ones( dis_real.size()), requires_grad=False)
    labels_dis_fake = Variable(torch.zeros(dis_fake.size()), requires_grad=False)
    labels_gen      = Variable(torch.ones( dis_fake.size()), requires_grad=False)

    if cuda:
        labels_dis_real = labels_dis_real.cuda()
        labels_dis_fake = labels_dis_fake.cuda()
        labels_gen = labels_gen.cuda()

    dis_loss = criterion( dis_real, labels_dis_real ) * 0.5 + criterion( dis_fake, labels_dis_fake ) * 0.5
    gen_loss = criterion( dis_fake, labels_gen )

    return dis_loss, gen_loss


def main():
    global args
    args = parser.parse_args()

    cuda = args.cuda
    if cuda == 'true':
        cuda = True
    else:
        cuda = False

    task_name = args.task_name

    epoch_size = args.epoch_size
    batch_size = args.batch_size

    # path
    result_path = os.path.join( args.result_path, args.task_name )
    result_path = os.path.join( result_path, args.model_arch )
    model_path = os.path.join( args.model_path, args.task_name )
    model_path = os.path.join( model_path, args.model_arch )
    stats_path = os.path.join( model_path, "stats.json" )
    if not os.path.exists(result_path):
        os.makedirs(result_path)
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    json.dump({"data": []}, open(stats_path, "w"))
    stats = json.load(open(stats_path))["data"]

    # dataset
    data_style_A, data_style_B, test_style_A, test_style_B = get_data()
    test_A = read_images( test_style_A, None, args.image_size, aspect=args.no_aspect, invert=args.no_invert )
    test_B = read_images( test_style_B, None, args.image_size, aspect=args.no_aspect, invert=args.no_invert )
    test_A = Variable( torch.FloatTensor( test_A ), requires_grad=False)
    test_B = Variable( torch.FloatTensor( test_B ), requires_grad=False)

    # tmp: save training data
    train_A = read_images( data_style_A, None, args.image_size, aspect=args.no_aspect, invert=args.no_invert )
    train_B = read_images( data_style_B, None, args.image_size, aspect=args.no_aspect, invert=args.no_invert )
    train_A = Variable( torch.FloatTensor( train_A ), requires_grad=False)
    train_B = Variable( torch.FloatTensor( train_B ), requires_grad=False)

    generator_A = Generator()
    generator_B = Generator()
    discriminator_A = Discriminator()
    discriminator_B = Discriminator()

    if cuda:
        test_A = test_A.cuda()
        test_B = test_B.cuda()
        generator_A = generator_A.cuda()
        generator_B = generator_B.cuda()
        discriminator_A = discriminator_A.cuda()
        discriminator_B = discriminator_B.cuda()

    data_size = min( len(data_style_A), len(data_style_B) )
    n_batches = ( data_size // batch_size )

    recon_criterion = nn.MSELoss()
    recon_criterion_ssim = pytorch_ssim.SSIM(window_size=11)
    gan_criterion = nn.BCELoss()

    gen_params = chain(generator_A.parameters(), generator_B.parameters())
    dis_params = chain(discriminator_A.parameters(), discriminator_B.parameters())

    optim_gen = optim.Adam( gen_params, lr=args.learning_rate, betas=(0.5,0.999), weight_decay=0.00001)
    optim_dis = optim.Adam( dis_params, lr=args.learning_rate, betas=(0.5,0.999), weight_decay=0.00001)

    iters = 0

    gen_loss_total = []
    dis_loss_total = []

    for epoch in range(epoch_size):
        # data_style_A, data_style_B = shuffle_data( data_style_A, data_style_B)
        # shuffle
        a_idx = list(range(len(train_A)))
        np.random.shuffle(a_idx)
        b_idx = list(range(len(train_B)))
        np.random.shuffle(b_idx)

        widgets = ['epoch #%d|' % epoch, Percentage(), Bar(), ETA()]
        pbar = ProgressBar(maxval=n_batches, widgets=widgets)
        pbar.start()

        for i in range(n_batches):

            pbar.update(i)

            generator_A.zero_grad()
            generator_B.zero_grad()
            discriminator_A.zero_grad()
            discriminator_B.zero_grad()

            # A_path = data_style_A[ i * batch_size: (i+1) * batch_size ]
            # B_path = data_style_B[ i * batch_size: (i+1) * batch_size ]
            # A = read_images( A_path, None, args.image_size )
            # B = read_images( B_path, None, args.image_size )
            # A = Variable( torch.FloatTensor( A ) )
            # B = Variable( torch.FloatTensor( B ) )

            A = train_A[a_idx[ i * batch_size: (i+1) * batch_size ]]
            B = train_B[b_idx[ i * batch_size: (i+1) * batch_size ]]

            if cuda:
                A = A.cuda()
                B = B.cuda()

            AB = generator_B(A)
            BA = generator_A(B)

            ABA = generator_A(AB)
            BAB = generator_B(BA)

            # Reconstruction Loss
            recon_loss_A = recon_criterion( ABA, A ) * .2 + (1 - recon_criterion_ssim( ABA, A )) * .8
            recon_loss_B = recon_criterion( ABA, A ) * .2 + (1 - recon_criterion_ssim( BAB, B )) * .8

            # Real/Fake GAN Loss (A)
            A_dis_real = discriminator_A( A )
            A_dis_fake = discriminator_A( BA )

            dis_loss_A, gen_loss_A = get_gan_loss( A_dis_real, A_dis_fake, gan_criterion, cuda )

            # Real/Fake GAN Loss (B)
            B_dis_real = discriminator_B( B )
            B_dis_fake = discriminator_B( AB )

            dis_loss_B, gen_loss_B = get_gan_loss( B_dis_real, B_dis_fake, gan_criterion, cuda )

            # Total Loss

            if iters < args.gan_curriculum:
                rate = args.starting_rate
            else:
                rate = args.default_rate

            gen_loss_A_total = gen_loss_B * (1.-rate) + recon_loss_A * rate
            gen_loss_B_total = gen_loss_A * (1.-rate) + recon_loss_B * rate

            if args.model_arch == 'discogan':
                gen_loss = gen_loss_A_total + gen_loss_B_total
                dis_loss = dis_loss_A + dis_loss_B
            elif args.model_arch == 'recongan':
                gen_loss = gen_loss_A_total
                dis_loss = dis_loss_B
            elif args.model_arch == 'gan':
                gen_loss = gen_loss_B
                dis_loss = dis_loss_B

            if iters % args.update_interval == 0:
                dis_loss.backward()
                optim_dis.step()
            else:
                gen_loss.backward()
                optim_gen.step()

            with torch.no_grad():
                if iters % args.log_interval == 0:
                    stat = {
                        "iter": iters,
                        "genlossA":   float(gen_loss_A.mean()),
                        "genlossB":   float(gen_loss_B.mean()),
                        "reconlossA": float(recon_loss_A.mean()),
                        "reconlossB": float(recon_loss_B.mean()),
                        "dislossA":   float(dis_loss_A.mean()),
                        "dislossB":   float(dis_loss_B.mean()),
                    }
                    print("---------------------")
                    print("GEN Loss:", stat["genlossA"], stat["genlossB"])
                    print("RECON Loss:", stat["reconlossA"], stat["reconlossB"])
                    print("DIS Loss:", stat["dislossA"], stat["dislossB"])
                    stats.append(stat)
                    json.dump({"data": stats}, open(stats_path, "w"))

                if iters % args.image_save_interval == 0:
                    AB = generator_B( test_A )
                    BA = generator_A( test_B )
                    ABA = generator_A( AB )
                    BAB = generator_B( BA )

                    n_testset = min( test_A.size()[0], test_B.size()[0] )

                    subdir_path = os.path.join( result_path, str(iters / args.image_save_interval) )

                    if os.path.exists( subdir_path ):
                        pass
                    else:
                        os.makedirs( subdir_path )

                    for im_idx in range( n_testset ):
                        A_val   = as_np(test_A[im_idx]).transpose(1,2,0) * 255.
                        B_val   = as_np(test_B[im_idx]).transpose(1,2,0) * 255.
                        BA_val  = as_np(    BA[im_idx]).transpose(1,2,0) * 255.
                        ABA_val = as_np(   ABA[im_idx]).transpose(1,2,0) * 255.
                        AB_val  = as_np(    AB[im_idx]).transpose(1,2,0) * 255.
                        BAB_val = as_np(   BAB[im_idx]).transpose(1,2,0) * 255.

                        filename_prefix = os.path.join (subdir_path, str(im_idx))
                        Image.fromarray(  A_val.astype(np.uint8)[:,:,::-1]).save( filename_prefix + '.A.jpg')
                        Image.fromarray(  B_val.astype(np.uint8)[:,:,::-1]).save( filename_prefix + '.B.jpg')
                        Image.fromarray( BA_val.astype(np.uint8)[:,:,::-1]).save( filename_prefix + '.BA.jpg')
                        Image.fromarray( AB_val.astype(np.uint8)[:,:,::-1]).save( filename_prefix + '.AB.jpg')
                        Image.fromarray(ABA_val.astype(np.uint8)[:,:,::-1]).save( filename_prefix + '.ABA.jpg')
                        Image.fromarray(BAB_val.astype(np.uint8)[:,:,::-1]).save( filename_prefix + '.BAB.jpg')

                if iters % args.model_save_interval == 0:
                    torch.save( generator_A,     os.path.join(model_path, 'model_gen_A-' + str( iters / args.model_save_interval )))
                    torch.save( generator_B,     os.path.join(model_path, 'model_gen_B-' + str( iters / args.model_save_interval )))
                    torch.save( discriminator_A, os.path.join(model_path, 'model_dis_A-' + str( iters / args.model_save_interval )))
                    torch.save( discriminator_B, os.path.join(model_path, 'model_dis_B-' + str( iters / args.model_save_interval )))

            iters += 1

if __name__=="__main__":
    main()
