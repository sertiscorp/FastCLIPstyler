import numpy as np

import torch
import torch.nn as nn

class PBNModel():
    def __init__(self):

        checkpoint_embeddings = torch.load('styleaug/checkpoints/pbn_embeddings.pth')

        self.means_ = checkpoint_embeddings['pbn_embedding_mean'].type(torch.float32)
        self.covariances_ =  checkpoint_embeddings['pbn_embedding_covariance'].type(torch.float32)
        self.precisions_ =  torch.inverse(self.covariances_)

    def sample(self, n_samples):
        pass

class PBNEmbedding(nn.Module):
    def __init__(self):

        super(PBNEmbedding, self).__init__()

        self.device = None

        self.pbn_model = PBNModel()

        self.means = self.pbn_model.means_
        self.cov = self.pbn_model.covariances_
        self.cov_inv = self.pbn_model.precisions_

        u, s, vh = np.linalg.svd(self.cov.numpy())
        self.A = np.matmul(u, np.diag(s ** 0.5))
        self.A = torch.Tensor(self.A)

    def get_latent_dim(self):
        return self.means.shape[0]

    def sample(self, n_samples=100):

        embeddings = torch.randn(n_samples, self.means.shape[0]).to(self.device)
        embeddings = torch.mm(embeddings, 1.5*self.A.transpose(1, 0)) + self.means

        return embeddings

    def compute_proba(self, x):

        x_centered = x-self.means
        cov_inv = self.cov_inv

        probs = torch.einsum('ij,jk,ik->i', x_centered, cov_inv, x_centered)

        return probs.mean()

    def to(self, device):
        self.means = self.means.to(device)
        self.cov = self.cov.to(device)
        self.cov_inv = self.cov_inv.to(device)
        self.A = self.A.to(device)

        self.device = device


    def requires_grad_(self, requires_grad):
        self.means.requires_grad_(requires_grad)
        self.cov.requires_grad_(requires_grad)
        self.cov_inv.requires_grad_(requires_grad)
        self.A.requires_grad_(requires_grad)