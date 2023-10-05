import numpy as np
import torch
import gpytorch
import tqdm
import seaborn as sns
from matplotlib import pyplot as plt

plt.rcParams['figure.figsize'] = [16, 8]

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

class SpectralMixtureGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(SpectralMixtureGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.SpectralMixtureKernel(num_mixtures=4)
        self.covar_module.initialize_from_data(train_x, train_y)

    def forward(self,x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

class MultitaskGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(MultitaskGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.MultitaskMean(
                gpytorch.means.ConstantMean(), num_tasks=4
                )
        self.covar_module = gpytorch.kernels.MultitaskKernel(
                gpytorch.kernels.RBFKernel(), num_tasks=4, rank=4
                )

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultitaskMultivariateNormal(mean_x, covar_x)


def train(model, likelihood, training_iter=1000):
    # Use the Adam optmizer  
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer=optimizer,
            gamma=0.99,
            verbose=False
            )
    # "Loss" for GPs - the marginal log likelihood
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

    iterator = tqdm.trange(training_iter)
    loss_array = np.empty(shape=(training_iter, ))
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
        scheduler.step()
        iterator.set_description(f"Loss: {loss}")
        loss_array[i] = loss

    plt.plot(loss_array)
    plt.savefig("./figures/learning_curve.png")
    plt.close()

# Make predictions with the model
def predict(model, likelihood, test_x):
    model.eval()
    likelihood.eval()
    # Make predictions by feeding model through likelihood
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        return likelihood(model(test_x))


def plot(model, likelihood, test_x, test_y):
    # Compute the Prior distribution
    y_prior = likelihood(model.forward(test_x))
    # Train the Model 
    model.train()
    likelihood.train()
    train(model, likelihood)
    # Compute the Posterior distribution
    y_posterior = predict(model, likelihood, test_x)
    # Plot the Model
    with torch.no_grad():
        # Initialize plot
        f, ax = plt.subplots(2, 3)
        # Get upper and lower confidence bounds
        lower, upper = y_prior.confidence_region()
        # Plot pior means as blue line
        ax[0, 0].plot(test_x.numpy(), y_prior.mean.numpy(), 'b')
        # Shade between the lower and upper confidence bounds
        ax[0, 0].fill_between(test_x.numpy(), lower.numpy(), upper.numpy(), alpha=0.5)
        ax[0, 0].legend(['Mean', 'Confidence'])

        sns.heatmap(y_prior.covariance_matrix.numpy(), ax=ax[0, 1])
        ax[0, 2].plot(test_x.numpy(), y_prior.sample(torch.Size([5])).T)

        lower, upper = y_posterior.confidence_region()
        # Plot training data as black stars
        ax[1, 0].plot(train_x.numpy(), train_y.numpy(), 'k*')
        # Plot testing data as red stars
        ax[1, 0].plot(test_x.numpy(), test_y.numpy(), 'r*')
        # Plot predictive means as blue line
        ax[1, 0].plot(test_x.numpy(), y_posterior.mean.numpy(), 'b')
        # Shade between the lower and upper confidence bounds
        ax[1, 0].fill_between(test_x.numpy(), lower.numpy(), upper.numpy(), alpha=0.5)
        ax[1, 0].legend(['Training Data', 'Testing Data', 'Mean', 'Confidence'])

        sns.heatmap(y_posterior.covariance_matrix.numpy(), ax=ax[1, 1])
        ax[1, 2].plot(test_x.numpy(), y_posterior.sample(torch.Size([5])).T)
        plt.savefig("./figures/gp_plots.png")

    init_nlpd = gpytorch.metrics.negative_log_predictive_density(y_prior, test_y)
    final_nlpd = gpytorch.metrics.negative_log_predictive_density(y_posterior, test_y)

    print(f'Untrained model NLPD: {init_nlpd:.2f}, \nTrained model NLPD: {final_nlpd:.2f}')

    init_msll = gpytorch.metrics.mean_standardized_log_loss(y_prior, test_y)
    final_msll = gpytorch.metrics.mean_standardized_log_loss(y_posterior, test_y)

    print(f'Untrained model MSLL: {init_msll:.2f}, \nTrained model MSLL: {final_msll:.2f}')
    
    init_mse = gpytorch.metrics.mean_squared_error(y_prior, test_y, squared=True)
    final_mse = gpytorch.metrics.mean_squared_error(y_posterior, test_y, squared=True)

    print(f'Untrained model MSE: {init_mse:.2f}, \nTrained model MSE: {final_mse:.2f}')

    init_mae = gpytorch.metrics.mean_absolute_error(y_prior, test_y)
    final_mae = gpytorch.metrics.mean_absolute_error(y_posterior, test_y)

    print(f'Untrained model MAE: {init_mae:.2f}, \nTrained model MAE: {final_mae:.2f}')

def correlation_from_covariance(covariance):
    v = np.sqrt(np.diag(covariance))
    outer_v = np.outer(v, v)
    correlation = covariance / outer_v
    correlation[covariance == 0] = 0
    return correlation
# Set Up training data 

def y(x):
    return torch.stack([
        torch.sin(x * 2 * torch.pi) + torch.randn(x.size()) * np.sqrt(0.04),
        torch.sign(torch.sin(x * 4 * torch.pi)) + torch.randn(x.size()) * np.sqrt(0.04),
        x + torch.rand(x.size()) * np.sqrt(0.04),
        1 + torch.rand(x.size()) * np.sqrt(0.04),
        ], -1)
train_x = torch.linspace(0, 0.6, 50)
train_y = y(train_x)
test_x = torch.linspace(0, 1, 100)
test_y = y(test_x)
# The likelihood
likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks=4)
model = MultitaskGPModel(train_x, train_y, likelihood)
# model = SpectralMixtureGPModel(train_x, train_y, likelihood)
train(model, likelihood)

# Set into eval mode
model.eval()
likelihood.eval()
# Make predictions
with torch.no_grad(), gpytorch.settings.fast_pred_var():
    predictions = likelihood(model(test_x))
    mean = predictions.mean
    lower, upper = predictions.confidence_region()
# Number of tasks
n_tasks = 4  # Change this to the number of tasks you have

# Calculate the number of rows and columns for the subplot grid
n_rows = int(n_tasks ** 0.5)
n_cols = n_tasks // n_rows + (n_tasks % n_rows > 0)

# Initialize plots with squared subplots
fig, axes = plt.subplots(n_rows, n_cols, figsize=(10, 10))

# Flatten the axes array for easier iteration
axes = axes.flatten()

# Iterate over tasks
for i in range(n_tasks):
    ax = axes[i]

    # Plot training data as black stars
    ax.plot(train_x.detach().numpy(), train_y[:, i].detach().numpy(), 'k*')
    ax.plot(test_x.detach().numpy(), test_y[:, i].detach().numpy(), 'r*')
    
    # Predictive mean as blue line
    ax.plot(test_x.numpy(), mean[:, i].numpy(), 'b')
    
    # Shade in confidence
    ax.fill_between(test_x.numpy(), lower[:, i].numpy(), upper[:, i].numpy(), alpha=0.5)
    
    ax.legend(['Training Data', 'Testing Data', 'Mean', 'Confidence'])
    ax.set_title(f'Observed Values (Task {i+1})')

# Remove any empty subplots if n_tasks is not a perfect square
for i in range(n_tasks, n_rows * n_cols):
    fig.delaxes(axes[i])

plt.tight_layout()
plt.show()

H = model.covar_module.task_covar_module.covar_factor.detach().numpy()

sns.heatmap(correlation_from_covariance(H @ H.T), annot=True)
plt.show()


# Save The Model
torch.save(model.state_dict(), './models/model_state.pth')


