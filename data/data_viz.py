import os
import time
import torch
import argparse
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from torchvision import transforms
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
from collections import defaultdict
import pdb 

from data_MNIST import MNIST, trans_col_MNIST, trans_MNIST, add_conf, data_lab_MNIST 
from models import VAE

def main(args):

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ts = time.time()

    # if train == True: 
    index_n_labels, index_n_labels_v =  data_lab_MNIST(split='test')
    labels_conf = add_conf(index_n_labels,p=0.5,qyu=0.90,N=1500)

    dataset = MNIST(labels = labels_conf, conf=0, conf_type='colour',
                                transform = trans_col_MNIST, data_ty='test')

    data_loader = DataLoader(dataset, batch_size=100, shuffle=True)

    vae = VAE(
        encoder_layer_sizes=args.encoder_layer_sizes,
        latent_size=args.latent_size,
        decoder_layer_sizes=args.decoder_layer_sizes,
        conditional=args.conditional,
        num_labels=10 if args.conditional else 0).to(device)

    # checkpoint = torch.load('./models/best_model_20.pth')
    checkpoint = torch.load('./models/best_model_30_condt.pth')
    vae.load_state_dict(checkpoint['state_dict'])
    
    torch.set_grad_enabled(False)

    for p in range(10):

        pdb.set_trace()

        if args.conditional:
            # c = torch.arange(0, 10).long().unsqueeze(1).to(device)
            c = p*torch.ones(10).long().to(device)
            z = torch.randn([c.size(0), args.latent_size]).to(device)
            x = vae.inference(z, c=c)
        else:
            z = torch.randn([10, args.latent_size]).to(device)
            x = vae.inference(z)

        plt.figure()
        plt.figure(figsize=(5, 10))

        pdb.set_trace()

        for j in range(10): 

            plt.subplot(5, 2, j+1)
            if args.conditional:
                plt.text(
                    0, 0, "c = {:d}".format(p), color='black',
                    backgroundcolor='white', fontsize=8)
            # sko.imsave('test.png', x)
            plt.imshow(x[j].permute(1,2,0).cpu().data.numpy())
            plt.axis('off')

        if not os.path.exists(os.path.join(args.fig_path, 'vis_30_cond')):
                if not(os.path.exists(os.path.join(args.fig_path))):
                    os.mkdir(os.path.join(args.fig_path))
                os.mkdir(os.path.join(args.fig_path, 'vis_30_cond'))

        plt.savefig(os.path.join(args.fig_path,'vis_30_cond',
                                    "it - {:d}.png".format(p)),
                                    dpi=300)
        plt.clf()
        plt.close('all')

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--seed",'-s', type=int, default=0)
    parser.add_argument("--epochs",'-e', type=int, default=10)
    parser.add_argument("--batch_size",'-b', type=int, default=64)
    parser.add_argument("--learning_rate", type=float, default=0.001)
    parser.add_argument("--encoder_layer_sizes", type=list, default=[784,256,100,50])
    parser.add_argument("--decoder_layer_sizes", type=list, default=[50,100,256,784])
    parser.add_argument("--latent_size", '-ls',type=int, default=120)
    parser.add_argument("--print_every", type=int, default=100)
    parser.add_argument("--fig_path",'-f',type=str, default='figs')
    parser.add_argument("--model_path",'-m', type=str, default='model')
    parser.add_argument("--conditional", '-c', action='store_true')
    parser.add_argument("--interventional", '-i', action='store_true')

    args = parser.parse_args()

    main(args)
