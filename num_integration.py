import numpy as np
from scipy.stats import gamma, poisson
from scipy.integrate import simpson
import matplotlib.pyplot as plt

lambdas = np.linspace(0, 10, 1_000)
alpha_0, beta_0 = 2, 1
data = [5]

# conjugate solution
alpha, beta = alpha_0 + sum(data), beta_0 + len(data)
posterior_conjugate = gamma.pdf(lambdas, a=alpha, scale=1/beta)

prior = gamma.pdf(lambdas, a=alpha_0, scale=1/beta_0)
likelihood = poisson.pmf(data, mu=lambdas)
numerator = prior * likelihood

marginal = simpson(y=numerator, x=lambdas)
posterior = numerator / marginal


fig = plt.figure(figsize=(6,4))

plt.plot(lambdas, prior, label='prior')
plt.plot(lambdas, posterior_conjugate, label='posterior (conjugate)')
plt.plot(lambdas, numerator, label='posterior (unscaled)')
plt.plot(lambdas, posterior, label='posterior (scaled)', linestyle='dotted')

plt.legend()
plt.show()



