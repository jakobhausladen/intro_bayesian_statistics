import numpy as np
import pandas as pd
from scipy.stats import norm, gamma
import matplotlib.pyplot as plt
from tqdm import tqdm

# Hyperparameters for prior distributions
mu_0_0, tau_0_0 = 0.0, 0.0001
mu_1_0, tau_1_0 = 0.0, 0.0001
alpha_0, beta_0 = 0.01, 0.01

# Data
data = pd.DataFrame({
    'X': [12, 14, 18, 10, 13, 22, 17, 15, 16, 9, 19, 8, 20, 11, 21],
    'Y': [33.48, 42.53, 48.53, 30.21, 38.76, 38.59, 52.93, 32.65, 52.42, 22.22, 41.4, 16.28, 40.83, 24.43, 56.38]
})

# Initial values for the sampler
b_0_p = 6.0
b_1_p = 0.3
tau_p = None

def sample_tau(b_0_p, b_1_p):
    """
    Sample tau from its conditional posterior distribution.
    """
    alpha_p = alpha_0 + (len(data) / 2)
    ss = np.sum(np.square(data['Y'] - b_0_p - (b_1_p * data['X'])))
    beta_p = beta_0 + (ss / 2)
    return gamma.rvs(a=alpha_p, scale=1/beta_p)

def sample_b_0(b_1_p, tau_p):
    """
    Sample b_0 from its conditional posterior distribution.
    """
    num = (tau_0_0 * mu_0_0) + (tau_p * np.sum(data['Y'] - (b_1_p * data['X'])))
    denom = tau_0_0 + (len(data) * tau_p)
    mu_0_p = num / denom
    tau_0_p = denom
    return norm.rvs(loc=mu_0_p, scale=np.sqrt(1/tau_0_p))

def sample_b_1(b_0_p, tau_p):
    """
    Sample b_1 from its conditional posterior distribution.
    """
    num = (tau_1_0 * mu_1_0) + (tau_p * np.sum(data['X'] * (data['Y'] - b_0_p)))
    denom = tau_1_0 + (tau_p * np.sum(np.square(data['X'])))
    mu_1_p = num / denom
    tau_1_p = denom
    return norm.rvs(loc=mu_1_p, scale=np.sqrt(1/tau_1_p))

# Storage for samples
tau_samples = []
b_0_samples = []
b_1_samples = []
x_range = np.arange(8, 23)
predictions = []

# Gibbs sampling
for _ in tqdm(range(20_000)):
    tau_p = sample_tau(b_0_p, b_1_p)
    tau_samples.append(tau_p)
    b_0_p = sample_b_0(b_1_p, tau_p)
    b_0_samples.append(b_0_p)
    b_1_p = sample_b_1(b_0_p, tau_p)
    b_1_samples.append(b_1_p)
    y_pred = b_0_p + b_1_p * x_range
    predictions.append(y_pred)

# Calculate quantiles
predictions = np.array(predictions)
quantile_025 = np.quantile(predictions, 0.025, axis=0)
quantile_5 = np.quantile(predictions, 0.5, axis=0)
quantile_975 = np.quantile(predictions, 0.975, axis=0)

# Calculate summary statistics for the parameters
b_0_mean = np.mean(b_0_samples)
b_0_std = np.std(b_0_samples)
b_1_mean = np.mean(b_1_samples)
b_1_std = np.std(b_1_samples)
tau_mean = np.mean(tau_samples)
tau_std = np.std(tau_samples)

# Plot results
fig, axs = plt.subplots(2, 2, figsize=(10, 6))

# Regression plot
axs[0, 0].scatter(data['X'], data['Y'])
axs[0, 0].plot(x_range, quantile_025, color='blue', linestyle='dashed')
axs[0, 0].plot(x_range, quantile_5, color='blue')
axs[0, 0].plot(x_range, quantile_975, color='blue', linestyle='dashed')
axs[0, 0].set_ylim(0, 70)
axs[0, 0].text(0.05, 0.95, f'$Y \\sim \\mathcal{{N}}(b_0 + b_1 X, \\tau^{{-1}})$', transform=axs[0, 0].transAxes, fontsize=12, verticalalignment='top')
axs[0, 0].set_xlabel('X')
axs[0, 0].set_ylabel('Y')
axs[0, 0].set_title('Bayesian Linear Regression with Gibbs Sampling')

# Distribution of b_0
axs[0, 1].hist(b_0_samples, bins=80, density=True, color='blue', alpha=0.7)
axs[0, 1].set_title('Distribution of $b_0$')
axs[0, 1].text(0.95, 0.95, f"Mean = {b_0_mean:.2f}\nSD = {b_0_std:.2f}", 
               horizontalalignment='right', verticalalignment='top', transform=axs[0, 1].transAxes, fontsize=12)

# Distribution of b_1
axs[1, 0].hist(b_1_samples, bins=80, density=True, color='green', alpha=0.7)
axs[1, 0].set_title('Distribution of $b_1$')
axs[1, 0].text(0.95, 0.95, f"Mean = {b_1_mean:.2f}\nSD = {b_1_std:.2f}", 
               horizontalalignment='right', verticalalignment='top', transform=axs[1, 0].transAxes, fontsize=12)

# Distribution of tau
axs[1, 1].hist(tau_samples, bins=80, density=True, color='red', alpha=0.7)
axs[1, 1].set_title('Distribution of $\\tau$')
axs[1, 1].text(0.95, 0.95, f"Mean = {tau_mean:.2f}\nSD = {tau_std:.2f}", 
               horizontalalignment='right', verticalalignment='top', transform=axs[1, 1].transAxes, fontsize=12)

plt.tight_layout()
plt.show()
