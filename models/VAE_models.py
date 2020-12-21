import torch
import torch.nn as nn
import pdb 

from utils import idx2onehot

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

        # pdb.set_trace()
        # if x.dim() > 2:
        #     x = x.view(-1, 28*28)
        # pdb.set_trace()

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

class Encoder_Z(nn.Module):

    def __init__(self, layer_sizes, latent_size, conditional, num_labels):

        super().__init__()

        self.conditional = conditional
        self.num_labels = num_labels 

        # for i, (in_size, out_size) in enumerate(zip(layer_sizes[:-1], layer_sizes[1:])):
        #     self.MLP.add_module(
        #         name="L{:d}".format(i), module=nn.Linear(in_size, out_size))
        #     self.MLP.add_module(name="A{:d}".format(i), module=nn.ReLU())

        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 4, 2),
            nn.LeakyReLU(0.2),
            nn.Conv2d(32, 64, 4, 2),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, 4, 2),
            nn.LeakyReLU(0.2),
            # nn.Conv2d(128, 256, 4, 2),
            # nn.LeakyReLU(0.2),
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
                # , 
                # nn.Dropout(dropout)
            ])
        
        # last layer
        layers.extend([ 
                nn.Linear(layer_sizes[-1],hidden_size)
        ])

        self.fc = nn.Sequential(*layers) 
        # self.fc2 = nn.Linear(hidden_size, hidden_size)
        # self.fc1 = nn.Linear(hidden_size, hidden_size - num_labels)
        self.linear_means = nn.Linear(hidden_size, latent_size)
        self.linear_log_var = nn.Linear(hidden_size, latent_size)

    def forward(self, x, c=None):

        x = self.encoder(x)
        if self.conditional:
            c = idx2onehot(c, n=self.num_labels)
            x = torch.cat((x, c), dim=-1)
        x = self.fc(x)

        means = self.linear_means(x)
        log_vars = self.linear_log_var(x)

        return means, log_vars

class Decoder_X(nn.Module):

    def __init__(self, layer_sizes, latent_size, conditional, num_labels):

        super().__init__()

        # self.MLP = nn.Sequential()
        self.num_labels = num_labels 
       
        self.conditional = conditional
        if self.conditional:
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
            # nn.ConvTranspose2d(hidden_size, 128, 5, 2),
            # nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 5, 2),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 5, 2),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 3, 4, 2),
            nn.Sigmoid(),
        ) 

        # for i, (in_size, out_size) in enumerate(zip([input_size]+layer_sizes[:-1], layer_sizes)):
        #     self.MLP.add_module(
        #         name="L{:d}".format(i), module=nn.Linear(in_size, out_size))
        #     if i+1 < len(layer_sizes):
        #         self.MLP.add_module(name="A{:d}".format(i), module=nn.ReLU())
        #     else:
        #         self.MLP.add_module(name="sigmoid", module=nn.Sigmoid())

    def forward(self, z, c):

        # pdb.set_trace()

        if self.conditional:
            c = idx2onehot(c, n=self.num_labels)
            z = torch.cat((z, c), dim=-1)

        z = self.fc(z)
        x = self.decoder(z)

        return x

class Decoder_Y(nn.Module):
    def __init__(self, dim_in, n_class=2):
        super().__init__()
        self.fcs = nn.Sequential(
            nn.Linear(dim_in, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, n_class)
        )

    def forward(self, z):
        out = F.softmax(self.fcs(z), dim=-1)
        return out

class Encoder_W(nn.Module):

    def __init__(self, layer_sizes, latent_size, conditional, num_labels):

            super().__init__()

            self.conditional = conditional
            self.num_labels = num_labels 

            # for i, (in_size, out_size) in enumerate(zip(layer_sizes[:-1], layer_sizes[1:])):
            #     self.MLP.add_module(
            #         name="L{:d}".format(i), module=nn.Linear(in_size, out_size))
            #     self.MLP.add_module(name="A{:d}".format(i), module=nn.ReLU())

            self.encoder = nn.Sequential(
                nn.Conv2d(3, 32, 4, 2),
                nn.LeakyReLU(0.2),
                nn.Conv2d(32, 64, 4, 2),
                nn.LeakyReLU(0.2),
                nn.Conv2d(64, 128, 4, 2),
                nn.LeakyReLU(0.2),
                # nn.Conv2d(128, 256, 4, 2),
                # nn.LeakyReLU(0.2),
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
                    # , 
                    # nn.Dropout(dropout)
                ])
            
            # last layer
            layers.extend([ 
                    nn.Linear(layer_sizes[-1],hidden_size)
            ])

            self.fc = nn.Sequential(*layers) 
            # self.fc2 = nn.Linear(hidden_size, hidden_size)
            # self.fc1 = nn.Linear(hidden_size, hidden_size - num_labels)
            self.linear_means = nn.Linear(hidden_size, latent_size)
            self.linear_log_var = nn.Linear(hidden_size, latent_size)

        def forward(self, x, c=None):

            x = self.encoder(x)
            if self.conditional:
                c = idx2onehot(c, n=self.num_labels)
                x = torch.cat((x, c), dim=-1)
            x = self.fc(x)

            means = self.linear_means(x)
            log_vars = self.linear_log_var(x)

            return means, log_vars

        def __init__(self, dim_in, dim_out):
            super().__init__()
            self.dim_out = dim_out
            self.fcs = nn.Sequential(
                nn.Linear(dim_in, 64),
                nn.ReLU(),
                nn.Linear(64, 64),
                nn.ReLU(),
                nn.Linear(64, 2 * dim_out)
            )

    def forward(self, x, y):
        out = self.fcs(torch.cat((x, y), dim=-1))
        mu = out[:, :self.dim_out]
        logvar = out[:, self.dim_out:]  # decorrelated gaussian
        sample = torch.randn_like(mu) * torch.exp(0.5 * logvar) + mu
        return mu, logvar, sample
