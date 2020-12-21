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
from utils import tile
import pdb 
from data.data_MNIST import MNIST, trans_col_MNIST, trans_MNIST, add_conf, data_lab_MNIST 
from models.VAE_models import VAE

def main(args):

    torch.manual_seed(args.seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    ts = time.time()

    index_n_labels, index_n_labels_v =  data_lab_MNIST(split='train')    
    labels_conf = add_conf(index_n_labels,p=0.5,qyu=0.90,N=50000)

    dataset = MNIST(labels = labels_conf, conf=0, conf_type='colour',
                                transform = trans_col_MNIST, data_ty='training', per_digit=False)
    
    data_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    def KL(mu1, logvar1, mu2, logvar2):
        std1 = torch.exp(0.5 * logvar1)
        std2 = torch.exp(0.5 * logvar2)
        KL_div = torch.sum(torch.log(std2) - torch.log(std1) + (0.5*(torch.exp(logvar1) + (mu1 - mu2).pow(2)) / torch.exp(logvar2)) - 0.5) 

        return KL_div

    


    optimizer1 = torch.optim.Adam(chain(decoder_x.parameters(), 
                encoder_w.parameters(), encoder_z.parameters()), lr=1e-3)
    optimizer2 = torch.optim.Adam(decoder_y.parameters(), lr=1e-3)
    logs = defaultdict(list)

    for epoch in range(args.epochs):

        tracker_epoch = defaultdict(lambda: defaultdict(dict))

        for iteration, (x, y) in enumerate(data_loader):

            x, y = x.to(device), y.to(device)

            mu_z, logvar_z, z = encoder_z(x)
            mu_w, logvar_w, w = encoder_w(x, y.unsqueeze(-1).float())

            mu_x, logvar_x, pred_x = decoder_x(w, z)
            pred_y = decoder_y(z)

            kl1 = KL(mu_w, logvar_w, torch.zeros_like(mu_w), torch.ones_like(logvar_w) * np.log(0.01))
            kl0 = KL(mu_w, logvar_w, torch.ones_like(mu_w) * 3., torch.zeros_like(logvar_w))

            optimizer1.zero_grad()
            loss1 = ( torch.sum((x - mu_x) ** 2, -1)
                    + torch.where(y == 1, kl1, kl0)
                    + KL(mu_z, logvar_z, torch.zeros_like(mu_z), torch.zeros_like(logvar_z))
                    + torch.sum(pred_y * torch.log(pred_y), -1)).sum()  # maximize entropy, enforce uniform distribution
            loss1.backward(retain_graph=True)
            optimizer1.step()
                
            optimizer2.zero_grad()
            loss2 = (100. * torch.where(y == 1, -torch.log(pred_y[:, 1]), -torch.log(pred_y[:, 0]))).sum()
            loss2.backward()
            optimizer2.step()

            loss = loss1 + loss2
            logs['loss'].append(loss.item())

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
               
                    ax_1.imshow(recon_x[p].permute(1,2,0).cpu().data.numpy())
                    ax_1.axis('off')

                os.makedirs(os.path.join(args.fig_root, str(ts)), exist_ok=True)

                plt.savefig(
                    os.path.join(args.fig_root, str(ts),
                                "org{:d}I{:d}.png".format(epoch, iteration)),
                    dpi=300)
                plt.clf()
                plt.close('all')

                with torch.no_grad():
                    zs, _, _ = encoder_z(torch.from_numpy(xs))
                    ws, _, _ = encoder_w(torch.from_numpy(xs), torch.from_numpy(ys).unsqueeze(-1).float())
                    pred_x, _, _ = decoder_x(ws, zs)
                    pred_x = pred_x.cpu().numpy()
                
                if args.conditional:
                    y = torch.arange(0, 10).long().unsqueeze(1).to(device)
                    z = torch.randn([c.size(0), args.latent_size]).to(device)
                    x_recon = vae.inference(z, c=c)
                else:
                    z = torch.randn([10, args.latent_size]).to(device)
                    x = vae.inference(z)
                
                plt.figure()
                plt.figure(figsize=(5, 10))
                
                # for i,p in enumerate([0,1]):
                for p in range(10):
                    # plt.subplot(5, 2, i+1)
                    plt.subplot(5, 2, p+1)
                    if args.conditional:
                        plt.text(
                            0, 0, "c={:d}".format(c[p].item()), color='black',
                            backgroundcolor='white', fontsize=8)
                    plt.imshow(x[p].permute(1,2,0).cpu().data.numpy())
                    plt.axis('off')

                if not os.path.exists(os.path.join(args.fig_root, str(ts))):
                    if not(os.path.exists(os.path.join(args.fig_root))):
                        os.mkdir(os.path.join(args.fig_root))
                    os.mkdir(os.path.join(args.fig_root, str(ts)))

                plt.savefig(
                    os.path.join(args.fig_root, str(ts),
                                 "E{:d}I{:d}.png".format(epoch, iteration)),
                    dpi=300)
                plt.clf()
                plt.close('all')

        t.set_postfix(loss=loss.item(), y_max=pred_y.max().item(), y_min=pred_y.min().item())
    
    
    


    torch.save({
                'state_dict': vae.state_dict()
            }, f'{args.model_path}')

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--seed",'-s', type=int, default=0)
    parser.add_argument("--epochs",'-e', type=int, default=10)
    parser.add_argument("--batch_size",'-b', type=int, default=64)
    parser.add_argument("--learning_rate", type=float, default=0.001)
    parser.add_argument("--encoder_layer_sizes", type=list, default=[784,256,100,50])
    parser.add_argument("--decoder_layer_sizes", type=list, default=[50,100,256,784])
    parser.add_argument("--latent_size",'-ls',type=int, default=10)
    parser.add_argument("--print_every", type=int, default=100)
    parser.add_argument("--fig_root",'-f', type=str, default='figs')
    parser.add_argument("--model_path",'-m', type=str, default='model')
    parser.add_argument("--conditional", '-c', action='store_true')
    parser.add_argument("--interventional", '-i', action='store_true')
    parser.add_argument("--consub", '-cs', action='store_true')

    args = parser.parse_args()

    main(args)
