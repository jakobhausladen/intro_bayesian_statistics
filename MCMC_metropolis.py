import numpy as np
from scipy.stats import norm, uniform, beta, gamma, poisson
from tqdm import tqdm
import matplotlib.pyplot as plt
from typing import Callable, Tuple, Optional, Union, Dict
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class MCMC:
    def __init__(self, starting_point: float, proposal_scale: float, random_state: Optional[int] = None):
        self.starting_point = starting_point
        self.proposal_scale = proposal_scale
        self.prior_func = None
        self.likelihood_func = None
        self.current = None
        self.sampled = None
        self.burn_in = None
        
        # Set random state for reproducibility
        if random_state is not None:
            np.random.seed(random_state)

    def set_prior(self, distribution: Callable, fixed_params: Dict[str, Union[float, int]]):
        """
        Set the prior distribution function with fixed parameters.
        """
        def prior(parameter: float) -> float:
            """
            Calculates the prior probability density of the parameter given the fixed distribution parameters.
            """
            if hasattr(distribution, 'pdf'):
                return distribution.pdf(parameter, **fixed_params)
            else:
                raise ValueError("Unsupported distribution type for prior")
        
        self.prior_func = prior

    def set_likelihood(self, distribution: Callable, data: np.ndarray, variable_param: str, fixed_params: Optional[Dict[str, Union[float, int]]] = None):
        """
        Set the likelihood function with fixed data and a variable parameter.
        """
        if fixed_params is None:
            fixed_params = {}
        
        def likelihood(parameter: float) -> float:
            """
            Calculates the likelihood density of the data (fixed) given the parameter (input).
            """
            all_params = {**fixed_params, variable_param: parameter}
            if hasattr(distribution, 'pmf'):
                return np.prod(distribution.pmf(data, **all_params))
            elif hasattr(distribution, 'pdf'):
                return np.prod(distribution.pdf(data, **all_params))
            else:
                raise ValueError("Unsupported distribution type for likelihood")
        
        self.likelihood_func = likelihood

    def _bayes_theorem(self, parameter: float) -> float:
        """
        Calculate the posterior probability density given a parameter value.
        """
        if not self.prior_func or not self.likelihood_func:
            raise RuntimeError("Prior and likelihood functions must be set before running the MCMC sampler.")

        likelihood = self.likelihood_func(parameter)
        prior = self.prior_func(parameter)
        return likelihood * prior

    def _sample_proposed(self) -> float:
        """
        Sample a proposed new value using a normal distribution centered at the current value.
        """
        return norm.rvs(loc=self.current, scale=self.proposal_scale)

    def _metropolis(self, proposed: float) -> Tuple[float, bool]:
        """
        Decide whether to accept or reject the proposed value using the Metropolis criterion.
        """
        density_current = self._bayes_theorem(self.current)
        density_proposed = self._bayes_theorem(proposed)

        p_move = min(density_proposed / density_current, 1)
        random = uniform.rvs(loc=0, scale=1)

        if random < p_move:
            self.current = proposed
            return self.current, True
        return self.current, False

    def run(self, n: int, burn_in: int = 0):
        """
        Run the MCMC sampling for n iterations.
        """
        if not self.prior_func or not self.likelihood_func:
            raise RuntimeError("Prior and likelihood functions must be set before running the MCMC sampler.")

        self.current = self.starting_point
        self.sampled = []
        accepted = []
        for _ in tqdm(range(n)):
            proposed = self._sample_proposed()
            self.current, moved = self._metropolis(proposed)
            accepted.append(moved)
            self.sampled.append(self.current)

        self.burn_in = burn_in
        acceptance_rate = sum(accepted[burn_in:]) / (n - burn_in)
        logging.info('Finished sampling.')
        logging.info('-----------------------------')
        logging.info(f'Number of trials: {n}')
        logging.info(f'Burn-in period: {burn_in}')
        logging.info(f'Acceptance rate: {acceptance_rate:.2f}')

    def plot(self, plot_chain: bool = True, figsize: Tuple[int, int] = (12, 4), bins: int = 40):
        """
        Plot the prior distribution and the histogram of the posterior samples and/or the Markov chain progression over trials.
        """
        fig = plt.figure(figsize=figsize)
        gs = fig.add_gridspec(1, 2, width_ratios=[1.5, 2])

        ax0 = fig.add_subplot(gs[0])
        ax1 = fig.add_subplot(gs[1])
        
        # Plot prior distribution
        x_values = np.linspace(0, 12, 200)
        y_values = self.prior_func(x_values)
        ax0.plot(x_values, y_values, label='Prior')

        # Plot posterior samples as histogram
        ax0.hist(self.sampled[self.burn_in:], bins=bins, density=True, alpha=0.5, label='Posterior')

        ax0.set_xlabel('Parameter Value')
        ax0.set_ylabel('Density')
        ax0.set_title('Prior and Posterior Distributions')
        ax0.legend()

        if plot_chain:
            ax1.plot(self.sampled[:self.burn_in], marker='o', linestyle='-', markersize=3, label='Burn-in')
            ax1.plot(range(self.burn_in, len(self.sampled)), self.sampled[self.burn_in:], marker='o', linestyle='-', markersize=3, label='Post burn-in')
            ax1.set_title('Markov Chain Progression')
            ax1.set_xlabel('Trial')
            ax1.set_ylabel('Parameter Value')
            ax1.legend()
        
        plt.tight_layout()
        plt.show()



if __name__ == "__main__":
    mcmc = MCMC(starting_point=20, proposal_scale=2, random_state=42)
    data = [4,6,7,8,4,6]    
    mcmc.set_prior(gamma, {'a': 2.1, 'scale':1})
    mcmc.set_likelihood(poisson, data, variable_param="mu")
    
    mcmc.run(n=10000, burn_in=100)
    mcmc.plot(bins=50)
