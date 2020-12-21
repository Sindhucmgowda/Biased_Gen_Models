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

    # og_dataset = MNIST(labels = labels_conf, conf=0, conf_type='colour',
    #                             transform = trans_col_MNIST, data_ty='training', per_digit=True)
    # og_data_loader = DataLoader(og_dataset, batch_size=10, shuffle=False)
    # x_og, y_og = next(iter(og_data_loader))

    # pdb.set_trace()

    def KL(mu1, logvar1, mu2, logvar2):
        std1 = torch.exp(0.5 * logvar1)
        std2 = torch.exp(0.5 * logvar2)
        KL_div = torch.sum(torch.log(std2) - torch.log(std1) + (0.5*(torch.exp(logvar1) + (mu1 - mu2).pow(2)) / torch.exp(logvar2)) - 0.5) 

        return KL_div

    def loss_fn(recon_x, x, mean, log_var, y):

        # Prior Means 
        # For binary images 
        # BCE = torch.nn.functional.binary_cross_entropy(
        # recon_x.view(-1, 28*28), x.view(-1, 28*28), reduction='sum') 
        # KLD_old = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())

        # For coloured 3 channel images
        if (args.conditional or args.interventional):
            BCE = torch.nn.functional.mse_loss(recon_x, x, reduction='sum')
            KLD = KL(mean, log_var, 10*y.repeat(args.latent_size,1).t(), torch.zeros_like(log_var)) 
    
        else: 
            pdb.set_trace()
            BCE = torch.nn.functional.mse_loss(recon_x, x, reduction='sum')
            KLD = KL(mean, log_var, torch.zeros_like(mean), torch.zeros_like(log_var)) 

        return (BCE + KLD) / x.size(0)

    vae = VAE(
        latent_size=args.latent_size,
        encoder_z_layer_sizes=args.encoder_layer_sizes,
        decoder_x_layer_sizes=args.decoder_layer_sizes,
        conditional=args.conditional,
        interventional=args.interventional
        num_labels=10 if (args.conditional or args.interventional) else 0, 
        ).to(device)

    optimizer = torch.optim.Adam(vae.parameters(), lr=args.learning_rate)
    logs = defaultdict(list)

    for epoch in range(args.epochs):

        tracker_epoch = defaultdict(lambda: defaultdict(dict))

        for iteration, (x, y) in enumerate(data_loader):

            x, y = x.to(device), y.to(device)

            if args.conditional:
                recon_x, mean, log_var, z = vae(x, y)
                loss = loss_fn(recon_x, x, mean, log_var, y)
            elif args.interventional: 
                recon_x_all, mean, log_var, z = vae(x, y)
                recon_x = recon_x_all[0]; recon_x_zall = recon_x_all[1] 
                x_og_rep = x_og.repeat(z.shape[0],1,1,1)
                
                x_new = torch.cat((x_og_rep, x),0)
                recon_x_new = torch.cat((recon_x_zall, recon_x),0)
                loss = loss_fn(recon_x_new, x_new, mean, log_var, y)
            else:
                recon_x, mean, log_var, z = vae(x)
                loss = loss_fn(recon_x, x, mean, log_var, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            logs['loss'].append(loss.item())

            plt.figure()
            plt.figure(figsize=(10, 10))

            if iteration % args.print_every == 0 or iteration == len(data_loader)-1:

                for p in range(10):
                    # plt.subplot(5, 2, i+1)
                    ax = plt.subplot2grid((10, 2), (p,0))
                    if args.conditional or args.interventional:
                        ax.text(
                            0, 0, "c={:d}".format(y[p].item()), color='black',
                            backgroundcolor='white', fontsize=8)
                    # sko.imsave('test.png', x)
                    ax.imshow(x[p].permute(1,2,0).cpu().data.numpy())
                    ax.axis('off')

                    ax_1 = plt.subplot2grid((10, 2), (p,1))
                    if args.conditional or args.interventional:
                        ax_1.text(
                            0, 0, "c={:d}".format(y[p].item()), color='black',
                            backgroundcolor='white', fontsize=8)
                    # sko.imsave('test.png', x)
                    ax_1.imshow(recon_x[p].permute(1,2,0).cpu().data.numpy())
                    ax_1.axis('off')

                if not os.path.exists(os.path.join(args.fig_root, str(ts))):
                    if not(os.path.exists(os.path.join(args.fig_root))):
                        os.mkdir(os.path.join(args.fig_root))
                    os.mkdir(os.path.join(args.fig_root, str(ts)))

                plt.savefig(
                    os.path.join(args.fig_root, str(ts),
                                    "org{:d}I{:d}.png".format(epoch, iteration)),
                    dpi=300)
                plt.clf()
                plt.close('all')

            for i, yi in enumerate(y):
                id = len(tracker_epoch)
                tracker_epoch[id]['x'] = z[i, 0].item()
                tracker_epoch[id]['y'] = z[i, 1].item()
                tracker_epoch[id]['label'] = yi.item()

            c_mean = torch.ones(args.latent_size) 
            # pdb.set_trace()

            if iteration % args.print_every == 0 or iteration == len(data_loader)-1:
                
                print("Epoch {:02d}/{:02d} Batch {:04d}/{:d}, Loss {:9.4f}".format(
                    epoch, args.epochs, iteration, len(data_loader)-1, loss.item()))

                if args.conditional:
                    c = torch.arange(0, 10).long().unsqueeze(1).to(device)
                    # pdb.set_trace()
                    # z = 10*c.repeat(1,args.latent_size) + torch.randn([c.size(0), args.latent_size]).to(device)
                    z = torch.randn([c.size(0), args.latent_size]).to(device)
                    x = vae.inference(z, c=c)
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

        df = pd.DataFrame.from_dict(tracker_epoch, orient='index')
        g = sns.lmplot(
            x='x', y='y', hue='label', data=df.groupby('label').head(100),
            fit_reg=False, legend=True)
        g.savefig(os.path.join(
            args.fig_root, str(ts), "E{:d}-Dist.png".format(epoch)),
            dpi=300)

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
