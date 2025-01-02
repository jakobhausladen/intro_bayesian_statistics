import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import poisson, binom


POISSON_RATE = 5
n_values = [6, 8, 10, 20]
x = np.arange(0, 15)

y = poisson.pmf(x, POISSON_RATE)
plt.plot(x, y, label=f'Poisson(lambda={POISSON_RATE})')

for n in n_values:
    p = round(POISSON_RATE / n, 3) # p*n = POISSON_RATE stays constant while n increases
    y = binom.pmf(x, n, p)
    plt.plot(x, y, linestyle='dashed', label=f'Binom(n={n}, p={p})')

plt.legend()
plt.show()