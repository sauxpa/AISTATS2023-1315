import numpy as np
from typing import Type, Union
from tqdm import tqdm
from .contextual_bandits import LinearBanditEnv
from .utils import inv_sherman_morrison


def random_lin_strategy(
    bandit: Type[LinearBanditEnv],
    T: int,
    regularization: float = 1.0,
    seed: Union[int, None] = None,
    verbose: bool = True,
):
    """Strategy that selects arms randomly at each turn for benchmark purpose.
    verbose: whether to display tqdm.
    """
    # Freeze seed
    bandit._seed(seed)

    K = bandit.n_arms
    d = bandit.dim_features
    assert T >= K, "Horizon should be higher than number of arms."

    visit_times = np.zeros(K)

    V_inv = 1 / regularization * np.eye(d * K)
    b = np.zeros(d * K)

    regrets = np.zeros(T)

    # Observe features
    features = bandit._gen_features()

    for t in tqdm(range(T), disable=not verbose):
        action = np.random.choice(K)
        # Pull arm
        _, reward, _, _ = bandit.step(action, features)

        # Update number of visit to arm
        visit_times[action] += 1

        # Update ridge regression
        A = np.zeros((d, K))
        A[:, action] = features[:, action]
        A = A.flatten(order="F")
        # V_inv = inv_woodbury(A, V_inv)
        V_inv = inv_sherman_morrison(A, V_inv)
        b += A * reward

        theta_hat = V_inv @ b

        # Observe next features
        features = bandit._gen_features()

        mus_hat = np.zeros(K)
        for k in bandit.arms:
            A = np.zeros((d, K))
            A[:, k] = features[:, k]
            A = A.flatten(order="F")
            mus_hat[k] = theta_hat @ A

        regrets[t] = bandit.mu_star(features) - bandit.mus(features)[action]

    return regrets


def constant_lin_strategy(
    bandit: Type[LinearBanditEnv],
    T: int,
    action: int = 0,
    regularization: float = 1.0,
    seed: Union[int, None] = None,
    verbose: bool = True,
):
    """Strategy that constantly selects the same arm
    at each turn for benchmark purpose.
    verbose: whether to display tqdm.
    """
    # Freeze seed
    bandit._seed(seed)

    K = bandit.n_arms
    d = bandit.dim_features
    assert T >= K, "Horizon should be higher than number of arms."
    assert action in range(K), "Unavailable action."

    visit_times = np.zeros(K)

    V_inv = 1 / regularization * np.eye(d * K)
    b = np.zeros(d * K)

    regrets = np.array(T)

    # Observe features
    features = bandit._gen_features()

    for t in tqdm(range(T), disable=not verbose):
        # Pull arm
        _, reward, _, _ = bandit.step(action, features)

        # Update number of visit to arm
        visit_times[action] += 1

        # Update ridge regression
        A = np.zeros((d, K))
        A[:, action] = features[:, action]
        A = A.flatten(order="F")
        # V_inv = inv_woodbury(A, V_inv)
        V_inv = inv_sherman_morrison(A, V_inv)
        b += A * reward

        theta_hat = V_inv @ b

        # Observe next features
        features = bandit._gen_features()

        mus_hat = np.zeros(K)
        for k in bandit.arms:
            A = np.zeros((d, K))
            A[:, k] = features[:, k]
            A = A.flatten(order="F")
            mus_hat[k] = theta_hat @ A

        regrets[t] = bandit.mu_star(features) - bandit.mus(features)[action]

    return regrets
