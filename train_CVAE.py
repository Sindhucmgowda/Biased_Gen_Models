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
import numpy as np 

from utils import tile, idx2onehot
import pdb 
import torch.nn.functional as F 
from data.data_MNIST import MNIST, trans_col_MNIST, trans_MNIST, add_conf, data_lab_MNIST 
from models.VAE_models import *

def main(args):

    torch.manual_seed(args.seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    ts = time.time()

    index_n_labels, index_n_labels_v =  data_lab_MNIST(split='train')    
    labels_conf = add_conf(index_n_labels,p=0.5,qyu=0.90,N=45000)

    dataset = MNIST(labels = labels_conf, conf=0, conf_type='colour',
                                transform = trans_col_MNIST, data_ty='training', per_digit=False)
    
    data_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    def KL(mu1, logvar1, mu2, logvar2, weights):

        std1 = torch.exp(0.5 * logvar1)
        std2 = torch.exp(0.5 * logvar2)
        KL_div = torch.sum(weights*(torch.log(std2) - torch.log(std1) + (0.5*(torch.exp(logvar1) + (mu1 - mu2).pow(2)) / torch.exp(logvar2)) - 0.5) ) 

        return KL_div

    encoder_z = Encoder_Z(args.encoder_lsizes_z, args.latent_size_z, conditional=False, num_labels=10).to(device)
    encoder_w = Encoder_W(args.encoder_lsizes_w, args.latent_size_w, conditional=True, num_labels=10).to(device)
        
    ## decoders 
    decoder_x = Decoder_X(
            args.decoder_lsizes_x, args.latent_size_z, False, True, args.latent_size_w).to(device)
    decoder_y_z = Decoder_Y(
            args.decoder_lsizes_y, args.latent_size_z, num_labels=10).to(device)

    ## trying this out to see if I can use the weights p(y/w) to learn unbiased model  
    decoder_y_w = Decoder_Y(
            args.decoder_lsizes_y, args.latent_size_w, num_labels=10).to(device)

    optimizer1 = torch.optim.Adam(list(decoder_x.parameters()) + list( 
                encoder_w.parameters()) +list(encoder_z.parameters()), lr=1e-3)
    optimizer2 = torch.optim.Adam(decoder_y_z.parameters(), lr=1e-3)
    optimizer3 = torch.optim.Adam(decoder_y_w.parameters(), lr=1e-3)    
    
    logs = defaultdict(list)

    for epoch in range(args.epochs):

        tracker_epoch = defaultdict(lambda: defaultdict(dict))

        for iteration, (x, y) in enumerate(data_loader):

            x, y = x.to(device), y.to(device)

            mu_z, logvar_z, z = encoder_z(x) ## so this is finding p(z/x) and not p(z/x,y) like in the paper ... ? 
            mu_w, logvar_w, w = encoder_w(x, y)

            pred_x = decoder_x(z, w)
            pred_y = decoder_y_z(z)
            pred_y_w = decoder_y_w(w)

            y_onehot = idx2onehot(y, n=10)
            w_bk = ((y_onehot*pred_y_w).sum(dim=1)).pow(-1).unsqueeze(-1)

            # pdb.set_trace()

            kl_y = KL(mu_w, logvar_w, y.unsqueeze(-1)*torch.ones_like(mu_w), torch.zeros_like(logvar_w), weights=w_bk)
            
            optimizer1.zero_grad()
            mse = torch.nn.functional.mse_loss(pred_x, x, reduction='none')
            loss1 = ( 20. *(w_bk* torch.sum(mse.view(mse.shape[0],-1),dim=-1)).sum()
                    + kl_y
                    + 0.2 * KL(mu_z, logvar_z, torch.zeros_like(mu_z), torch.zeros_like(logvar_z), weights=w_bk)
                    + 1000. * torch.sum((w_bk*(pred_y * torch.log(pred_y))), -1).sum() ) # maximize entropy, enforce uniform distribution in predicting y from z 
            loss1.backward(retain_graph=True)
            optimizer1.step()
                
            optimizer2.zero_grad()
            loss2 = 100 * (y_onehot * -torch.log(pred_y)).sum()
            # loss2 = (100. * torch.where(y == 1, -torch.log(pred_y[:, 1]), -torch.log(pred_y[:, 0]))).sum()
            loss2.backward()
            optimizer2.step()

            loss3 = F.nll_loss(pred_y_w,y)
            optimizer3.zero_grad()
            loss3.backward()
            optimizer3.step()

            loss = loss1 + loss2
            logs['loss'].append(loss.item())
            logs['counts'].append(epoch*int(len(data_loader)/args.batch_size) + iteration)

            print("Epoch {:02d}/{:02d} Batch {:04d}/{:d}, Loss {:9.4f}".format(
                    epoch, args.epochs, iteration, len(data_loader)-1, loss.item()))
        
            if iteration % args.print_every == 0 or iteration == len(data_loader)-1:

                plt.figure()
                plt.figure(figsize=(10, 10))

                for p in range(10):
                    ax = plt.subplot2grid((10, 2), (p,0))
                    if args.conditional or args.interventional:
                        ax.text(
                            0, 0, "c={:d}".format(y[p].item()), color='black',
                            backgroundcolor='white', fontsize=8)
                    ax.imshow(x[p].permute(1,2,0).cpu().data.numpy())
                    ax.axis('off')

                    ax_1 = plt.subplot2grid((10, 2), (p,1))
                    if args.conditional or args.interventional:
                        ax_1.text(
                            0, 0, "c={:d}".format(y[p].item()), color='black',
                            backgroundcolor='white', fontsize=8)
               
                    ax_1.imshow(pred_x[p].permute(1,2,0).cpu().data.numpy())
                    ax_1.axis('off')

                os.makedirs(os.path.join(args.fig_root, str(ts)), exist_ok=True)

                plt.savefig(
                    os.path.join(args.fig_root, str(ts),
                                "org{:d}I{:d}.png".format(epoch, iteration)),
                    dpi=300)
                plt.clf()
                plt.close('all')

                with torch.no_grad():
                    c = torch.arange(0, 10).long().unsqueeze(1).to(device)
                    sam_z = torch.randn([c.size(0), args.latent_size_z]).to(device)
                    sam_w = (c*torch.ones([c.size(0), args.latent_size_w]) + torch.randn([c.size(0), args.latent_size_w])).to(device)
                    
                    ## checking if my w really is learning some bias -- does sampling it from the same mean for different classes give the same results?  
                    # sam_w = torch.zeros([c.size(0), args.latent_size_w]).to(device)

                    pred_x_sam = decoder_x(z,w)
                     
                plt.figure()
                plt.figure(figsize=(5, 10))
                
                for p in range(10):
                    plt.subplot(5, 2, p+1)
                    plt.text(0, 0, "c={:d}".format(c[p].item()), color='black',
                            backgroundcolor='white', fontsize=8)
                    plt.imshow(pred_x_sam[p].permute(1,2,0).cpu().data.numpy())
                    plt.axis('off')

                plt.savefig(
                    os.path.join(args.fig_root, str(ts),
                                 "E{:d}I{:d}.png".format(epoch, iteration)),
                    dpi=300)
                plt.clf()
                plt.close('all')
        
        plt.figure()
        plt.plot(logs['counts'],logs['loss'])
        plt.savefig(os.path.join(args.fig_root, str(ts),"loss"))
        plt.clf()

    os.makedirs(os.path.join('/scratch/gobi2/sindhu/gen/vae/', args.model_path), exist_ok=True)
    torch.save({
                'state_dict_decoder_x': decoder_x.state_dict(),
                'state_dict_decoder_y': decoder_y_z.state_dict(),
                'state_dict_encoder_w': encoder_w.state_dict(),
                'state_dict_encoder_z': encoder_z.state_dict(),
            }, f'{args.model_path}')

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--seed",'-s', type=int, default=0)
    parser.add_argument("--epochs",'-e', type=int, default=1)
    parser.add_argument("--batch_size",'-b', type=int, default=128)
    parser.add_argument("--learning_rate", type=float, default=0.001)
    parser.add_argument("--encoder_lsizes_z", type=list, default=[784,256,100,50])
    parser.add_argument("--decoder_lsizes_x", type=list, default=[50,100,256,784])
    
    parser.add_argument("--encoder_lsizes_w", type=list, default=[75,50])
    parser.add_argument("--decoder_lsizes_y", type=list, default=[50,100,256,784])
    
    parser.add_argument("--latent_size_z",'-lsz',type=int, default=10)
    parser.add_argument("--latent_size_w",'-lsw',type=int, default=2)

    parser.add_argument("--print_every", type=int, default=100)
    parser.add_argument("--fig_root",'-f', type=str, default='figs')
    parser.add_argument("--model_path",'-m', type=str, default='trail')
    parser.add_argument("--conditional", '-c', action='store_true')
    parser.add_argument("--interventional", '-i', action='store_true')
    parser.add_argument("--consub", '-cs', action='store_true')

    args = parser.parse_args()

    main(args)
