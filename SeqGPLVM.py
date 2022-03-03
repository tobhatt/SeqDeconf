import torch
import torch.nn as nn
import numpy as np

from helpers import KL_standard_normal, my_softplus


class SeqGPLVM(nn.Module):
    '''
    dim(A) = N x T
    dim(x) = N x T x (dim_cov + 1) (dim_cov for covariates and 1 for last treatment)
    dim(Z) = 1
    '''
    def __init__(self, x, A, z_init, GP_mapping, lr=1e-2, fixed_z=False, **kwargs):
        super(SeqcGPLVM, self).__init__()

        self.A = A
        self.x = x
        self.output_dim = A.size()[1] #Number of time steps, i.e., T

        if fixed_z:
            self.z_mu = z_init.clone()
            self.z_logsigma = -10.0 * torch.ones_like(z_init)
        else:
            self.z_mu = nn.Parameter(z_init.clone(), requires_grad=True)
            self.z_logsigma = nn.Parameter(-1.0 * torch.ones_like(z_init), requires_grad=True)

        A_means = A.mean(axis=(0,1))

        # for every output time step, create a separate GP object
        # GPs time-inhomogeneous
        self.GP_mappings = GP_mapping(intercept_init=A_means, **kwargs)

        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)

    def get_kernel_vars(self):
        return np.array([mapping.get_kernel_var().detach().numpy() for mapping in self.GP_mappings])

    def get_noise_sd(self):
        return np.array([mapping.get_noise_var().sqrt().detach().numpy() for mapping in self.GP_mappings])

    def get_lengthscales(self):
        return np.array([mapping.get_ls().detach().numpy() for mapping in self.GP_mappings])

    def get_z_inferred(self):
        return self.z_mu.detach().numpy()

    def sample_z(self):
        eps = torch.randn_like(self.z_mu)
        z = self.z_mu + my_softplus(self.z_logsigma) * eps
        return z

    def optimizer_step(self):

        loss = 0.0
        # sample z
        z = self.sample_z()
        for j in range(self.output_dim):
            loss += self.GP_mappings.total_loss(z, self.x[:, j:(j + 1), :].squeeze(), self.A[:, j:(j + 1)])

        # KL
        loss += KL_standard_normal(self.z_mu, my_softplus(self.z_logsigma))

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def predict(self, z_star, x_star, add_likelihood_variance=False, to_numpy=False):
        N_star = z_star.size()[0]
        f_mean = torch.zeros(N_star, self.output_dim)
        f_var = torch.zeros(N_star, self.output_dim)
        for j in range(self.output_dim):
            f_mean[:, j], f_var[:, j] = self.GP_mappings.predict(self.z_mu, self.x[:, j:(j + 1), :].squeeze(), self.A[:, j:(j + 1)], z_star, x_star[:, j:(j + 1), :].squeeze(), add_likelihood_variance)

        f_sd = torch.sqrt(1e-6 + f_var)

        if to_numpy:
            f_mean, f_sd = f_mean.detach().numpy(), f_sd.detach().numpy()

        return f_mean, f_sd

    def train(self, n_iter, verbose=200):

        for t in range(n_iter):

            loss = self.optimizer_step()

            if t % verbose == 0:
                print("Iter {0}. Loss {1}".format(t, loss))



