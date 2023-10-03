import numpy as np
import torch
import gpytorch
import tqdm
from matplotlib import pyplot as plt 

# Set Up training data 
train_x = torch.linspace(0, 1, 100)
train_y = torch.sin(train_x * 2 * np.pi) + torch.randn(train_x.size()) * np.sqrt(0.04)

# The GP Model
class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


# The likelihood
likelihood = gpytorch.likelihoods.GaussianLikelihood()
model = ExactGPModel(train_x, train_y, likelihood)

# Training the model
model.train()
likelihood.train()

optimizer = torch.optim.Adam(model.parameters(), lr=0.1)

mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

training_iter = 50
iterator = tqdm.trange(training_iter)
for i in iterator:
    # Zero all parameter gradients
    optimizer.zero_grad()
    # Call the model and compute the loss
    output = model(train_x)
    loss = -mll(output, train_y)
    # Call backward on the loss to fill in gradients
    loss.backward()
    # Take a step on the optimizer
    optimizer.step()
    iterator.set_description(f"Loss: {loss}")
