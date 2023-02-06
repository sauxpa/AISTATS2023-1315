import gym
from gym import spaces
import numpy as np
from typing import Callable, List, Tuple


class LinearBanditEnv(gym.Env):
    """
    Stationary linear contextual bandit environment.
    theta: (dim_features, n_arms)
    Rewards are generated as <theta, Psi_{kt}> + noise_t,
    where Psi_{kt} is the feature map at time t for arm k.
    """

    def __init__(
        self,
        theta: List[float] = [],
        features_dist: Callable = None,
        noise_dist: Callable = None,
    ):
        self._theta = np.array(theta)
        self._noise_dist = noise_dist
        self._features_dist = features_dist
        self._seed()

    @property
    def theta(self) -> List[float]:
        return self._theta

    @theta.setter
    def theta(self, new_theta: List[float]):
        self._theta = np.array(new_theta)

    @property
    def n_arms(self) -> int:
        return self.theta.shape[1]

    @property
    def arms(self) -> List[int]:
        return range(self.n_arms)

    @property
    def dim_features(self) -> int:
        return self.theta.shape[0]

    @property
    def noise_dist(self) -> Callable:
        return self._noise_dist

    @noise_dist.setter
    def noise_dist(self, new_noise_dist: Callable):
        self._noise_dist = new_noise_dist

    @property
    def _shared_noise(self) -> bool:
        return not isinstance(self.noise_dist, list)

    @property
    def features_dist(self) -> Callable:
        return self._features_dist

    @features_dist.setter
    def features_dist(self, new_features_dist: Callable):
        self._features_dist = new_features_dist

    def mus(self, features: List[float]) -> List[float]:
        return np.array([self.theta[:, k] @ features[:, k] for k in self.arms])

    def best_mean_arm(self, features: List[float]) -> int:
        return np.argmax(self.mus(features))

    def mu_star(self, features: List[float]) -> float:
        return np.max(self.mus(features))

    def arms_mean_gaps(self, features: List[float]) -> List[float]:
        return self.mu_star(features) - self.mus(features)

    @property
    def action_space(self):
        return spaces.Discrete(self.n_arms)

    def _gen_features(self) -> List[float]:
        return self.features_dist.rvs()

    def _seed(self, seed=None):
        if seed is not None:
            np.random.seed(seed)

    def step(
        self, action: int, features
    ) -> Tuple[List[float], float, bool, set]:  # noqa E501
        assert self.action_space.contains(action)

        if self._shared_noise:
            reward = self.theta[:, action] @ features[:, action] + self.noise_dist.rvs()
        else:
            reward = (
                self.theta[:, action] @ features[:, action]
                + self.noise_dist[action].rvs()
            )
        done = True

        return 0.0, reward, done, {}

    def reset(self):
        return 0

    def render(self, mode="human", close=False):
        pass
