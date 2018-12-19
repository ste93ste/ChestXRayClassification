import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init

from .vae import VariationalAutoencoder
from .vae import Encoder, Decoder

drop_rate = 0.2

class Classifier(nn.Module):
    def __init__(self, dims):
        """
        Single hidden layer classifier
        with softmax output.
        """
        super(Classifier, self).__init__()
        [x_dim, h_dim, y_dim, add] = dims
        self.add = add
        x_dim_new = 512 + self.add
        self.x_dim_new = x_dim_new
        #self.bnInput = nn.BatchNorm1d(225*225+self.add)
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

        self.dense1 = nn.Linear(x_dim_new, 128)
        self.bn1 = nn.BatchNorm1d(128)
        self.drop1 = nn.Dropout(drop_rate)

        self.dense2 = nn.Linear(128, 32)
        self.bn2 = nn.BatchNorm1d(32)
        self.drop2 = nn.Dropout(drop_rate)

        self.logits = nn.Linear(32, y_dim)

    def forward(self, x):
        x = x.view(-1, 225*225 + self.add)
        #x = self.bnInput(x)
        addValues = x[:, x.shape[1] - self.add : x.shape[1]]
        x = x[:, 0: x.shape[1] - self.add]
        x = x.view(-1, 1, 225, 225)
        x = self.dropconv1(self.bnconv1(F.relu((self.conv1(x)))))
        x = self.dropconv2(self.bnconv2(F.relu((self.conv2(x)))))
        x = self.dropconv3(self.bnconv3(F.relu((self.conv3(x)))))
        x = self.dropconv4(self.bnconv4(F.relu((self.conv4(x)))))
        x = self.dropconv5(self.bnconv5(F.relu((self.conv5(x)))))

        x = x.view(-1, self.x_dim_new - self.add)

        if len(addValues.size()) != 0:
                x = torch.cat([x, addValues], dim=1)  

        x = self.drop1(self.bn1(F.relu(self.dense1(x))))
        x = self.drop2(self.bn2(F.relu(self.dense2(x))))
        x = F.softmax(self.logits(x), dim=-1)
        return x

class DeepGenerativeModel(VariationalAutoencoder):
    def __init__(self, dims):
        """
        M2 code replication from the paper
        'Semi-Supervised Learning with Deep Generative Models'
        (Kingma 2014) in PyTorch.

        The "Generative semi-supervised model" is a probabilistic
        model that incorporates label information in both
        inference and generation.

        Initialise a new generative model
        :param dims: dimensions of x, y, z and hidden layers.
        """
        [x_dim, self.y_dim, z_dim, h_dim] = dims
        super(DeepGenerativeModel, self).__init__([x_dim, z_dim, h_dim])

        self.encoder = Encoder([x_dim, h_dim, z_dim, self.y_dim])
        self.decoder = Decoder([z_dim + self.y_dim, list(reversed(h_dim)), x_dim])
        self.classifier = Classifier([x_dim, h_dim[0], self.y_dim, 0])

        for m in self.modules():
            if isinstance(m, nn.Linear):
                init.xavier_normal_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x, y):
        # Add label and data and generate latent variable
        z, z_mu, z_log_var = self.encoder(torch.cat([x, y], dim=1))

        self.kl_divergence = self._kld(z, (z_mu, z_log_var))

        # Reconstruct data point from latent data and label
        x_mu = self.decoder(torch.cat([z, y], dim=1))

        return x_mu

    def classify(self, x):
        logits = self.classifier(x)
        return logits

    def sample(self, z, y):
        """
        Samples from the Decoder to generate an x.
        :param z: latent normal variable
        :param y: label (one-hot encoded)
        :return: x
        """
        y = y.float()
        x = self.decoder(torch.cat([z, y], dim=1))
        return x


class AuxiliaryDeepGenerativeModel(DeepGenerativeModel):
    def __init__(self, dims):
        """
        Auxiliary Deep Generative Models [Maaløe 2016]
        code replication. The ADGM introduces an additional
        latent variable 'a', which enables the model to fit
        more complex variational distributions.

        :param dims: dimensions of x, y, z, a and hidden layers.
        """
        [x_dim, y_dim, z_dim, a_dim, h_dim] = dims
        super(AuxiliaryDeepGenerativeModel, self).__init__([x_dim, y_dim, z_dim, h_dim])
        
        self.aux_encoder = Encoder([x_dim, h_dim, a_dim, 0])
        self.aux_decoder = Encoder([x_dim, h_dim, a_dim, z_dim + y_dim,])

        self.classifier = Classifier([x_dim, h_dim[0], y_dim, a_dim])

        self.encoder = Encoder([x_dim, h_dim, z_dim, a_dim + y_dim])
        self.decoder = Decoder([y_dim + z_dim, list(reversed(h_dim)), x_dim])

    def classify(self, x):
        # Auxiliary inference q(a|x)
        a, a_mu, a_log_var = self.aux_encoder(x)

        # Classification q(y|a,x)
        logits = self.classifier(torch.cat([x, a], dim=1))
        return logits

    def forward(self, x, y):
        """
        Forward through the model
        :param x: features
        :param y: labels
        :return: reconstruction
        """
        # Auxiliary inference q(a|x)
        q_a, q_a_mu, q_a_log_var = self.aux_encoder(x)

        # Latent inference q(z|a,y,x)
        z, z_mu, z_log_var = self.encoder(torch.cat([x, y, q_a], dim=1))

        # Generative p(x|z,y)
        x_mu, x_var = self.decoder(torch.cat([z, y], dim=1))

        # Generative p(a|z,y,x)
        p_a, p_a_mu, p_a_log_var = self.aux_decoder(torch.cat([x, y, z], dim=1))

        a_kl = self._kld(q_a, (q_a_mu, q_a_log_var), (p_a_mu, p_a_log_var))
        z_kl = self._kld(z, (z_mu, z_log_var))

        self.kl_divergence = a_kl + z_kl

        return x_mu, x_var, z

