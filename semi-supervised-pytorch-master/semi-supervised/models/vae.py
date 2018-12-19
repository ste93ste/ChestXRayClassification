import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn import init

from layers import GaussianSample, GaussianMerge, GumbelSoftmax
from inference import log_gaussian, log_standard_gaussian

image_size = 225
hidden_size = 512
drop_rate = 0.2

class Perceptron(nn.Module):
    def __init__(self, dims, activation_fn=F.relu, output_activation=None):
        super(Perceptron, self).__init__()
        self.dims = dims
        self.activation_fn = activation_fn
        self.output_activation = output_activation

        self.layers = nn.ModuleList(list(map(lambda d: nn.Linear(*d), list(zip(dims, dims[1:])))))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i == len(self.layers)-1 and self.output_activation is not None:
                x = self.output_activation(x)
            else:
                x = self.activation_fn(x)

        return x



class Encoder(nn.Module):
    def __init__(self, dims, sample_layer=GaussianSample):
        """
        Inference network

        Attempts to infer the probability distribution
        p(z|x) from the data by fitting a variational
        distribution q_φ(z|x). Returns the two parameters
        of the distribution (µ, log σ²).

        :param dims: dimensions of the networks
           given by the number of neurons on the form
           [input_dim, [hidden_dims], latent_dim].
        """
        super(Encoder, self).__init__()

        [x_dim, h_dim, z_dim, add] = dims
        self.add = add
        x_dim_new = hidden_size + self.add

        #self.bnInput = nn.BatchNorm1d(image_size*image_size+self.add)

        self.conv1 = nn.Conv2d(in_channels=1,
                               out_channels=32,
                               kernel_size=5,
                               stride=2)
        self.bnconv1 = nn.BatchNorm2d(32)
        self.dropconv1 = nn.Dropout2d(drop_rate)

        self.conv2 = nn.Conv2d(in_channels=32,
                               out_channels=32,
                               kernel_size=5,
                               stride=2)
        self.bnconv2 = nn.BatchNorm2d(32)
        self.dropconv2 = nn.Dropout2d(drop_rate)

        self.conv3 = nn.Conv2d(in_channels=32,
                               out_channels=32,
                               kernel_size=5,
                               stride=2)
        self.bnconv3 = nn.BatchNorm2d(32)
        self.dropconv3 = nn.Dropout2d(drop_rate)

        self.conv4 = nn.Conv2d(in_channels=32,
                               out_channels=32,
                               kernel_size=5,
                               stride=2)
        self.bnconv4 = nn.BatchNorm2d(32)
        self.dropconv4 = nn.Dropout2d(drop_rate)

        self.conv5 = nn.Conv2d(in_channels=32,
                               out_channels=32,
                               kernel_size=5,
                               stride=2)
        self.bnconv5 = nn.BatchNorm2d(32)
        self.dropconv5 = nn.Dropout2d(drop_rate)

        self.dense0 = nn.Linear(x_dim_new, 256)
        self.bn0 = nn.BatchNorm1d(256)
        self.drop0 = nn.Dropout(drop_rate)

        self.x_dim_new = x_dim_new
        self.sample = sample_layer(256, z_dim)

    def forward(self, x):
        x = x.view(-1, image_size*image_size + self.add)
        #x = self.bnInput(x)
        addValues = x[:, x.shape[1] - self.add : x.shape[1]]
        x = x[:, 0: x.shape[1] - self.add]
        x = x.view(-1, 1, image_size, image_size)
        x = self.dropconv1(self.bnconv1(F.relu((self.conv1(x)))))
        x = self.dropconv2(self.bnconv2(F.relu((self.conv2(x)))))
        x = self.dropconv3(self.bnconv3(F.relu((self.conv3(x)))))
        x = self.dropconv4(self.bnconv4(F.relu((self.conv4(x)))))
        x = self.dropconv5(self.bnconv5(F.relu((self.conv5(x)))))
        
        x = x.view(-1, self.x_dim_new - self.add)

        if len(addValues.size()) != 0:
                x = torch.cat([x, addValues], dim=1)  
             
        x = self.drop0(self.bn0(F.relu(self.dense0(x))))

        return self.sample(x)


class Decoder(nn.Module):
    def __init__(self, dims):
        """
        Generative network

        Generates samples from the original distribution
        p(x) by transforming a latent representation, e.g.
        by finding p_θ(x|z).

        :param dims: dimensions of the networks
            given by the number of neurons on the form
            [latent_dim, [hidden_dims], input_dim].
        """
        super(Decoder, self).__init__()

        [z_dim, h_dim, x_dim] = dims

        x_dim_new = 5408

        self.dense0 = nn.Linear(z_dim, 256)
        self.bn0 = nn.BatchNorm1d(256)
        self.drop0 = nn.Dropout(drop_rate)

        self.dense1 = nn.Linear(256, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.drop1 = nn.Dropout(drop_rate)

        self.dense2 = nn.Linear(512, 1024)
        self.bn2 = nn.BatchNorm1d(1024)
        self.drop2 = nn.Dropout(drop_rate)

        self.dense3 = nn.Linear(1024, 2048)
        self.bn3 = nn.BatchNorm1d(2048)
        self.drop3 = nn.Dropout(drop_rate)

        self.dense4 = nn.Linear(2048, x_dim_new)
        self.bn4 = nn.BatchNorm1d(x_dim_new)
        self.drop4 = nn.Dropout(drop_rate)

        self.deconv1 = nn.ConvTranspose2d(in_channels=32,
                               out_channels=32,
                               kernel_size=3,
                               stride=2)
        self.bndeconv1 = nn.BatchNorm2d(32)
        self.dropdeconv1 = nn.Dropout2d(drop_rate)

        self.deconv2 = nn.ConvTranspose2d(in_channels=32,
                               out_channels=32,
                               kernel_size=3,
                               stride=2)
        self.bndeconv2 = nn.BatchNorm2d(32)
        self.dropdeconv2 = nn.Dropout2d(drop_rate)

        self.deconv3 = nn.ConvTranspose2d(in_channels=32,
                               out_channels=32,
                               kernel_size=3,
                               stride=2)
        self.bndeconv3 = nn.BatchNorm2d(32)
        self.dropdeconv3 = nn.Dropout2d(drop_rate)

        self.mu = nn.ConvTranspose2d(in_channels=32,
                               out_channels=1,
                               kernel_size=5,
                               stride=2)

        self.var = nn.ConvTranspose2d(in_channels=32,
                               out_channels=1,
                               kernel_size=5,
                               stride=2)


    def forward(self, x):
        x = self.drop0(self.bn0(F.relu(self.dense0(x))))
        x = self.drop1(self.bn1(F.relu(self.dense1(x))))
        x = self.drop2(self.bn2(F.relu(self.dense2(x))))
        x = self.drop3(self.bn3(F.relu(self.dense3(x))))
        x = self.drop4(self.bn4(F.relu(self.dense4(x))))

        x = x.view(-1, 32, 13, 13)
        x = self.dropdeconv1(self.bndeconv1(F.relu(self.deconv1(x))))
        x = self.dropdeconv2(self.bndeconv2(F.relu(self.deconv2(x))))
        x = self.dropdeconv3(self.bndeconv3(F.relu(self.deconv3(x))))
        x_mu = self.mu(x)
        x_var = F.softplus(self.var(x))       
        x_mu = x_mu.view(-1, image_size*image_size)
        x_var = x_var.view(-1, image_size*image_size)

        return x_mu, x_var


class VariationalAutoencoder(nn.Module):
    def __init__(self, dims):
        """
        Variational Autoencoder [Kingma 2013] model
        consisting of an encoder/decoder pair for which
        a variational distribution is fitted to the
        encoder. Also known as the M1 model in [Kingma 2014].

        :param dims: x, z and hidden dimensions of the networks
        """
        super(VariationalAutoencoder, self).__init__()

        [x_dim, z_dim, h_dim] = dims
        self.z_dim = z_dim
        self.flow = None

        self.encoder = Encoder([x_dim, h_dim, z_dim, 0])
        self.decoder = Decoder([z_dim, list(reversed(h_dim)), x_dim])
        self.kl_divergence = 0

        for m in self.modules():
            if isinstance(m, nn.Linear):
                init.xavier_normal_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()

    def _kld(self, z, q_param, p_param=None):
        """
        Computes the KL-divergence of
        some element z.

        KL(q||p) = -∫ q(z) log [ p(z) / q(z) ]
                 = -E[log p(z) - log q(z)]

        :param z: sample from q-distribuion
        :param q_param: (mu, log_var) of the q-distribution
        :param p_param: (mu, log_var) of the p-distribution
        :return: KL(q||p)
        """
        (mu, log_var) = q_param

        if self.flow is not None:
            f_z, log_det_z = self.flow(z)
            qz = log_gaussian(z, mu, log_var) - sum(log_det_z)
            z = f_z
        else:
            qz = log_gaussian(z, mu, log_var)

        if p_param is None:
            pz = log_standard_gaussian(z)
        else:
            (mu, log_var) = p_param
            pz = log_gaussian(z, mu, log_var)

        kl = qz - pz

        return kl

    def add_flow(self, flow):
        self.flow = flow

    def forward(self, x, y=None):
        """
        Runs a data point through the model in order
        to provide its reconstruction and q distribution
        parameters.

        :param x: input data
        :return: reconstructed input
        """
        z, z_mu, z_log_var = self.encoder(x)

        self.kl_divergence = self._kld(z, (z_mu, z_log_var))

        x_mu = self.decoder(z)

        return x_mu

    def sample(self, z):
        """
        Given z ~ N(0, I) generates a sample from
        the learned distribution based on p_θ(x|z).
        :param z: (torch.autograd.Variable) Random normal variable
        :return: (torch.autograd.Variable) generated sample
        """
        return self.decoder(z)

