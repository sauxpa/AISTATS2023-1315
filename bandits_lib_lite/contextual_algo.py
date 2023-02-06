import numpy as np
from typing import Type, Union
from tqdm import tqdm
from .contextual_bandits import LinearBanditEnv
from .utils import inv_sherman_morrison


def LinUCB(
    bandit: Type[LinearBanditEnv],
    T: int,
    delta: float = 0.05,
    regularization: float = 1.0,
    seed: Union[int, None] = None,
    exploration_bonus_scale: float = 1.0,
    bound_theta: float = 1.0,
    bound_features: float = 1.0,
    warmup: int = 1,
    verbose: bool = True,
):
    """LinUCB algorithm for a fixed confidence level delta.
    exploration_bonus_scale: multiplicative factor in the
        UCB exploration bonus.
    verbose: whether to display tqdm.
    """
    # Freeze seed
    bandit._seed(seed)

    K = bandit.n_arms
    d = bandit.dim_features
    assert T >= K, "Horizon should be higher than number of arms."

    assert bound_theta >= 0, "bound_theta should be positive."
    assert bound_features > 0, "bound_features should be positive."

    visit_times = np.zeros(K)
    mus_hat = np.zeros(K)
    UCB = np.zeros(K)

    V_inv = 1 / regularization * np.eye(d * K)
    b = np.zeros(d * K)

    regrets = np.zeros(T)

    # Observe features
    features = bandit._gen_features()

    for t in tqdm(range(T), disable=not verbose):
        if t < K * warmup:
            # Round Robin
            action = t % K
        else:
            # Select action with higher upper bound
            action = np.argmax(UCB)

        # Pull arm
        _, reward, _, _ = bandit.step(action, features)

        # Update number of visit to arm
        visit_times[action] += 1

        # Update ridge regression
        A = np.zeros((d, K))
        A[:, action] = features[:, action]
        A = A.flatten(order="F")
        V_inv = inv_sherman_morrison(A, V_inv)
        b += A * reward

        # Observe next features
        features = bandit._gen_features()

        # Update exploration bonus
        beta = np.sqrt(
            regularization
        ) * bound_theta + exploration_bonus_scale * np.sqrt(
            2 * np.log(1 / delta)
            + d * np.log(regularization + (t - 1) * bound_features**2 / d)
        )

        theta_hat = V_inv @ b

        exploration_bonus = np.zeros(K)
        mus_hat = np.zeros(K)
        for k in bandit.arms:
            A = np.zeros((d, K))
            A[:, k] = features[:, k]
            A = A.flatten(order="F")
            exploration_bonus[k] = beta * np.sqrt(A.T @ V_inv @ A)
            mus_hat[k] = theta_hat @ A

        UCB = mus_hat + exploration_bonus
        
        regrets[t] = bandit.mu_star(features) - bandit.mus(features)[action]

    return regrets
