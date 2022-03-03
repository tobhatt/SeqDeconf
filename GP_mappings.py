import torch
import torch.nn as nn
import numpy as np

from torch.distributions.gamma import Gamma
from torch.distributions.normal import Normal

from helpers import my_softplus, grid_helper
from kernels import RBF, meanzeroRBF, addint_2D_kernel_decomposition, addint_kernel_diag


# GP with 4D ARD kernel on (z, x) - 3 covariates and one (past) treatment
class GP_multi_INT(nn.Module):

    def __init__(self, z_inducing=None, x_inducing=None, intercept_init=None):

        super().__init__()

        self.jitter = 1e-3
        self.dim_cov = x_inducing.shape[1]

        # kernel hyperparameters
        self.ls = nn.Parameter(2.0*torch.ones(self.dim_cov + 1), requires_grad=True)
        self.var = nn.Parameter(torch.zeros(1), requires_grad=True)
        self.noise_var = nn.Parameter(-1.0 * torch.ones(1), requires_grad=True)

        if intercept_init is None:
            self.intercept = nn.Parameter(torch.zeros(1), requires_grad=True)
        else:
            self.intercept = nn.Parameter(intercept_init, requires_grad=True)

        # create a grid of inducing points
        self.z_u, self.x_u = grid_helper(z_inducing, x_inducing)
        self.M = self.z_u.size()[0]

    def get_kernel_var(self):
        return 1e-4 + my_softplus(self.var)

    def get_ls(self):
        return 1e-4 + my_softplus(self.ls)

    def get_noise_var(self):
        return 1e-4 + my_softplus(self.noise_var)

    def get_K_without_noise(self, z, z2, x, x2, jitter=None):
        zx = torch.cat([z, x], dim=1)
        zx2 = torch.cat([z2, x2], dim=1)
        K = RBF(zx, zx2, self.get_ls(), self.get_kernel_var(), jitter)
        return K

    def log_prob(self, z, x, a):
        # inducing points log_prob
        subset = ~torch.isnan(a).reshape(-1)
        if subset.sum() > 0:
            a = a[subset, :]
            z = z[subset, :]
            x = x[subset, :]
        N = a.size()[0]
        a = (a - self.intercept)
        sigma2 = self.get_noise_var()
        sigma = torch.sqrt(sigma2)
        K_uu = self.get_K_without_noise(self.z_u, self.z_u, self.x_u, self.x_u, jitter=self.jitter)
        K_uf = self.get_K_without_noise(self.z_u, z, self.x_u, x)
        L = torch.cholesky(K_uu)
        A = torch.triangular_solve(K_uf, L, upper=False)[0] / sigma
        AAT = torch.mm(A, A.t())
        B = AAT + torch.eye(self.M)
        LB = torch.cholesky(B)
        Aerr = torch.mm(A, a)
        c = torch.triangular_solve(Aerr, LB, upper=False)[0] / sigma

        Kdiag = self.get_kernel_var().repeat(N)

        bound = -0.5 * N * np.log(2 * np.pi) - torch.sum(torch.log(torch.diag(LB))) - 0.5 * N * torch.log(
            sigma2) - 0.5 * torch.sum(torch.pow(a, 2)) / sigma2
        bound += 0.5 * torch.sum(torch.pow(c, 2)) - 0.5 * torch.sum(Kdiag) / sigma2 + 0.5 * torch.sum(torch.diag(AAT))
        return bound

    def predict(self, z, x, y, z_star, x_star, add_likelihood_variance=False):
        subset = ~torch.isnan(y).reshape(-1)
        if subset.sum() > 0:
            y = y[subset, :]
            z = z[subset, :]
            x = x[subset, :]
        Nstar = z_star.size()[0]
        y = (y - self.intercept)
        sigma2 = self.get_noise_var()
        sigma = torch.sqrt(sigma2)

        K_uu = self.get_K_without_noise(self.z_u, self.z_u, self.x_u, self.x_u, jitter=self.jitter)
        K_uf = self.get_K_without_noise(self.z_u, z, self.x_u, x)
        K_us = self.get_K_without_noise(self.z_u, z_star, self.x_u, x_star)

        L = torch.cholesky(K_uu)
        A = torch.triangular_solve(K_uf, L, upper=False)[0] / sigma
        AAT = torch.mm(A, A.t())
        B = AAT + torch.eye(self.M)
        LB = torch.cholesky(B)
        Aerr = torch.mm(A, y)
        c = torch.triangular_solve(Aerr, LB, upper=False)[0] / sigma

        tmp1 = torch.triangular_solve(K_us, L, upper=False)[0]
        tmp2 = torch.triangular_solve(tmp1, LB, upper=False)[0]
        mean = self.intercept + torch.mm(tmp2.t(), c)

        K_ss = self.get_K_without_noise(z_star, z_star, x_star, x_star)
        Kdiag = K_ss.diag()
        var = Kdiag + torch.pow(tmp2, 2).sum(dim=0) - torch.pow(tmp1, 2).sum(dim=0)

        if add_likelihood_variance:
            var += self.get_noise_var()

        return mean.reshape(-1), var

    def log_prob_fullrank(self, z, x, y):
        N = z.size()[0]
        y = (y - self.intercept).reshape(-1)
        K = self.get_K_without_noise(z, z, x, x, jitter=self.jitter, which_kernels=None)
        K_noise = self.get_noise_var() * torch.eye(N)
        return MultivariateNormal(torch.zeros_like(y), K + K_noise).log_prob(y)

    def prior_loss(self):
        prior_var = -Gamma(1.0, 1.0).log_prob(self.get_kernel_var()).sum()
        prior_ls = -Gamma(10.0, 1.0).log_prob(self.get_ls()).sum()
        return prior_var + prior_ls

    def total_loss(self, z, x, a):
        return -self.log_prob(z, x, a) + self.prior_loss()





