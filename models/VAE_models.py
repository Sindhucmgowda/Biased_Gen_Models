import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import idx2onehot

import pdb 

hidden_size = 128

class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)

class UnFlatten(nn.Module):
    def forward(self, input, size=256):
        return input.view(input.size(0), hidden_size, 1, 1)

class VAE(nn.Module):

    def __init__(self, latent_size, decoder_layer_sizes, encoder_layer_sizes,
                conditional=False, interventional=False, num_labels=0):

        super().__init__()

        if conditional or interventional:
            assert num_labels > 0
        assert type(encoder_layer_sizes) == list
        assert type(latent_size) == int
        assert type(decoder_layer_sizes) == list

        self.latent_size = latent_size
        self.interventional = interventional
        self.encoder_z = Encoder_Z(
            encoder_layer_sizes, latent_size, conditional, num_labels)
        self.decoder_x = Decoder_X(
            decoder_layer_sizes, latent_size, conditional, num_labels)

    def forward(self, x, c=None):

        means, log_var = self.encoder_z(x, c)
        z = self.reparameterize(means, log_var)
        recon_x = self.decoder_x(z, c)

        if self.interventional: 
            recon_x_zall = torch.zeros([10,x.shape[0],x.shape[1],x.shape[2],x.shape[3]])
            for y in range(10):
                recon_x_zall[y] = self.decoder(z, torch.tensor(y).repeat(x.shape[0]))   
            recon_x_zall = recon_x_zall.view([10*x.shape[0],x.shape[1],x.shape[2],x.shape[3]])
            recon_x = [recon_x_z, recon_x_zall] 
        
        return recon_x, means, log_var, z

    def reparameterize(self, mu, log_var):

        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)

        return mu + eps * std

    def inference(self, z, c=None):

        recon_x = self.decoder(z, c)

        return recon_x

# class CSVAE(nn.Module):

#     def __init__(self, latent_size, decoder_layer_sizes, encoder_layer_sizes,
#                 num_labels=0):

#         super().__init__()

#         encoder_lsizes_z = encoder_layer_sizes[0]
#         encoder_lsizes_w = encoder_layer_sizes[1]

#         decoder_lsizes_x = decoder_layer_sizes[0]
#         decoder_lsizes_y = decoder_layer_sizes[1]

#         assert type(encoder_layer_sizes) == list
#         assert type(latent_size) == int
#         assert type(decoder_layer_sizes) == list

#         self.latent_size_z = latent_size[0]
#         self.latent_size_w = latent_size[1]

#         ## encoders 
#         self.encoder_z = Encoder_Z(
#             encoder_lsizes_z, latent_size_z, conditional=False, num_labels)
#         self.encoder_w = Encoder_W(encoder_lsizes_w, latent_size_w, conditional=True, num_labels)
        
#         ##decoders 
#         self.decoder_x = Decoder_X(
#             decoder_lsizes_x, latent_size, conditional=True, num_labels)
#         self.decoder_x = Decoder_Y(
#             decoder_lsizes_y, latent_size_z, conditional=False, num_labels)

#     def forward(self, x, c=None):

#         means, log_var = self.encoder_z(x, c)
#         z = self.reparameterize(means, log_var)
#         recon_x = self.decoder_x(z, c)

#         if self.interventional: 
#             recon_x_zall = torch.zeros([10,x.shape[0],x.shape[1],x.shape[2],x.shape[3]])
#             for y in range(10):
#                 recon_x_zall[y] = self.decoder(z, torch.tensor(y).repeat(x.shape[0]))   
#             recon_x_zall = recon_x_zall.view([10*x.shape[0],x.shape[1],x.shape[2],x.shape[3]])
#             recon_x = [recon_x_z, recon_x_zall] 
        
#         return recon_x, means, log_var, z

#     def reparameterize(self, mu, log_var):

#         std = torch.exp(0.5 * log_var)
#         eps = torch.randn_like(std)

#         return mu + eps * std

#     def inference(self, z, c=None):

#         recon_x = self.decoder(z, c)

#         return recon_x

class Encoder_Z(nn.Module):

    def __init__(self, layer_sizes, latent_size, conditional, num_labels):

        super().__init__()

        self.conditional = conditional
        self.num_labels = num_labels 
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 4, 2),
            nn.LeakyReLU(0.2),
            nn.Conv2d(32, 64, 4, 2),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, 4, 2),
            nn.LeakyReLU(0.2),
            Flatten(),
        )
        
        layer_sizes[0] = hidden_size
        if self.conditional:
            layer_sizes[0] += num_labels
        
        layers = []
        for i in range(len(layer_sizes) -1):
            layers.extend([
                nn.Linear(layer_sizes[i], layer_sizes[i+1]),
                nn.BatchNorm1d(layer_sizes[i+1]),
                nn.ReLU()
            ])
        layers.extend([ 
                nn.Linear(layer_sizes[-1],int(hidden_size/4))
        ])

        self.fc = nn.Sequential(*layers) 
        self.linear_means = nn.Linear(int(hidden_size/4), latent_size)
        self.linear_log_var = nn.Linear(int(hidden_size/4), latent_size)

    def reparameterize(self, mu, log_var):

        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)

        return mu + eps * std

    def forward(self, x, c=None):

        x = self.encoder(x)
        if self.conditional:
            c = idx2onehot(c, n=self.num_labels)
            x = torch.cat((x, c), dim=-1)
        x = self.fc(x)

        means = self.linear_means(x)
        log_vars = self.linear_log_var(x)

        z_sam = self.reparameterize(means, log_vars)

        return means, log_vars, z_sam 

class Encoder_W(nn.Module):

    def __init__(self, layer_sizes, latent_size, conditional, num_labels):

            super().__init__()

            self.conditional = conditional
            self.num_labels = num_labels 

            self.encoder = nn.Sequential(
                nn.Conv2d(3, 32, 4, 2),
                nn.LeakyReLU(0.2),
                nn.Conv2d(32, 64, 4, 2),
                nn.LeakyReLU(0.2),
                nn.Conv2d(64, 128, 4, 2),
                nn.LeakyReLU(0.2),
                Flatten(),
            )
            
            layer_sizes[0] = hidden_size
            if self.conditional:
                layer_sizes[0] += num_labels

            layers = []
            for i in range(len(layer_sizes) -1):
                layers.extend([
                    nn.Linear(layer_sizes[i], layer_sizes[i+1]),
                    nn.BatchNorm1d(layer_sizes[i+1]),
                    nn.ReLU()
                ])
            # last layer
            layers.extend([ 
                    nn.Linear(layer_sizes[-1],int(hidden_size/4))
            ])

            self.fc = nn.Sequential(*layers) 
            self.linear_means = nn.Linear(int(hidden_size/4), latent_size)
            self.linear_log_var = nn.Linear(int(hidden_size/4), latent_size)

    def reparameterize(self, mu, log_var):

        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)

        return mu + eps * std

    def forward(self, x, c=None):

        x = self.encoder(x)
        c = idx2onehot(c, n=self.num_labels)
        x = torch.cat((x, c), dim=-1)
        x = self.fc(x)

        means = self.linear_means(x)
        log_vars = self.linear_log_var(x)

        w_sam = self.reparameterize(means, log_vars) 

        return means, log_vars, w_sam 
    
class Decoder_X(nn.Module):

    def __init__(self, layer_sizes, latent_size, conditional, subspace, num_labels):

        super().__init__()

        self.num_labels = num_labels 
        self.subspace = subspace 

        self.conditional = conditional
        if self.conditional or self.subspace: 
            layer_sizes[0] = latent_size + num_labels
        else:
            layer_sizes[0] = latent_size

        layers = []
        for i in range(len(layer_sizes) -1):
            layers.extend([
                nn.Linear(layer_sizes[i], layer_sizes[i+1]),
                nn.BatchNorm1d(layer_sizes[i+1]),
                nn.ReLU()
                # , 
                # nn.Dropout(dropout)
            ])
        
        # last layer
        layers.extend([ 
                nn.Linear(layer_sizes[-1],hidden_size)
        ])

        self.fc = nn.Sequential(*layers) 
        self.decoder = nn.Sequential(
            UnFlatten(),
            nn.ConvTranspose2d(128, 64, 5, 2),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 5, 2),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 3, 4, 2),
            nn.Sigmoid(),
        ) 

    def forward(self, z, c):

        if self.conditional:
            c = idx2onehot(c, n=self.num_labels)
            z = torch.cat((z, c), dim=-1)
        elif self.subspace:
            z = torch.cat((z, c), dim=-1)

        z = self.fc(z)
        x = self.decoder(z)

        return x

class Decoder_Y(nn.Module):

    def __init__(self, layer_sizes, latent_size, num_labels):

        super().__init__()

        self.num_labels = num_labels 
       
        layer_sizes[0] = latent_size
        layers = []
        for i in range(len(layer_sizes) -1):
            layers.extend([
                nn.Linear(layer_sizes[i], layer_sizes[i+1]),
                nn.BatchNorm1d(layer_sizes[i+1]),
                nn.ReLU()
            ])
        
        # last layer
        layers.extend([ 
                nn.Linear(layer_sizes[-1],num_labels)
        ])

        self.fc = nn.Sequential(*layers) 
    
    def forward(self, z):

        y = self.fc(z)
        y = F.softmax(y, dim=1)
        
        return y 