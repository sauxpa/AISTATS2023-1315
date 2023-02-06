import numpy as np
from scipy.optimize import minimize
from typing import Type, Union
from tqdm import tqdm
from .contextual_bandits import LinearBanditEnv


def ExpectileUCB(
    bandit: Type[LinearBanditEnv],
    T: int,
    q: float,
    delta: float = 0.05,
    regularization: float = 1.0,
    seed: Union[int, None] = None,
    exploration_bonus_scale: float = 1.0,
    bound_theta: float = 1.0,
    bound_features: float = 1.0,
    warmup: int = 1,
    verbose: bool = True,
):
    """ExpectileUCB with exact expectile calculation at each step
    for a fixed confidence level delta.
    exploration_bonus_scale: multiplicative factor in the
        UCB exploration bonus.
    verbose: whether to display tqdm.
    """
    # Freeze seed
    bandit._seed(seed)

    def expectile_loss(x: float) -> float:
        return (1 - q) * x**2 if x < 0 else q * x**2

    def expectile_loss_hessian(x: float) -> float:
        return 2 * (1 - q) if x < 0 else 2 * q

    m = 2 * np.minimum(q, 1 - q)
    M = 2 * np.maximum(q, 1 - q)
    kappa = M / m

    K = bandit.n_arms
    d = bandit.dim_features
    assert T >= K, "Horizon should be higher than number of arms."

    assert bound_theta >= 0, "bound_theta should be positive."
    assert bound_features > 0, "bound_features should be positive."

    visit_times = np.zeros(K)
    mus_hat = np.zeros(K)
    UCB = np.zeros(K)

    theta_hat = np.zeros((d * K))

    features_history = np.zeros((d * K, T))
    rewards_history = np.zeros(T)

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

        # Update expectile regression
        A = np.zeros((d, K))
        A[:, action] = features[:, action]
        A = A.flatten(order="F")
        features_history[:, t] = A
        rewards_history[t] = reward

        def empirical_expectile_loss(theta):
            return (
                np.sum(
                    [
                        expectile_loss(
                            rewards_history[s] - theta @ features_history[:, s]
                        )
                        for s in range(t + 1)
                    ]
                )
                + 0.5 * regularization * (theta**2).sum()
            )

        ret = minimize(
            empirical_expectile_loss,
            theta_hat,
        )

        if ret.success:
            theta_hat = ret.x
        # else:
        # raise Exception("Expectile calculation failed...")

        D2_loss = [
            expectile_loss_hessian(
                rewards_history[s] - theta_hat @ features_history[:, s]
            )
            for s in range(t + 1)
        ]

        H0 = np.sum(
            [
                D2_loss[s] * np.outer(features_history[:, s], features_history[:, s])
                for s in range(t + 1)
            ],
            axis=0,
        )

        H = H0 + kappa * regularization * np.eye(d * K)

        H_inv = np.linalg.inv(H)

        # Observe next features
        features = bandit._gen_features()

        # Update exploration bonus
        c = 2 * (
            kappa
            * exploration_bonus_scale
            * np.sqrt(
                2 * np.log(1 / delta)
                - d * K * np.log(kappa * regularization)
                + np.linalg.slogdet(H)[1]
            )
            + np.sqrt(kappa * regularization) * bound_theta
        )

        exploration_bonus = np.zeros(K)
        mus_hat = np.zeros(K)
        for k in bandit.arms:
            A = np.zeros((d, K))
            A[:, k] = features[:, k]
            A = A.flatten(order="F")
            exploration_bonus[k] = c * np.sqrt(A.T @ H_inv @ A)
            mus_hat[k] = theta_hat @ A

        UCB = mus_hat + exploration_bonus

        regrets[t] = bandit.mu_star(features) - bandit.mus(features)[action]

    return regrets


def ExpectileOGDUCB(
    bandit: Type[LinearBanditEnv],
    T: int,
    q: float,
    delta: float = 0.05,
    regularization: float = 1.0,
    step_size: float = 0.1,
    n_iter_ogd: int = 1,
    grad_clip: float = 10.0,
    seed: Union[int, None] = None,
    exploration_bonus_scale: float = 1.0,
    bound_theta: float = 1.0,
    bound_features: float = 1.0,
    warmup: int = 1,
    verbose: bool = True,
):
    """ExpectileOGDUCB with OGD expectile calculation at each step
    for a fixed confidence level delta.
    exploration_bonus_scale: multiplicative factor in the
        UCB exploration bonus.
    verbose: whether to display tqdm.
    """
    # Freeze seed
    bandit._seed(seed)

    def expectile_loss(x: float) -> float:
        return (1 - q) * x**2 if x < 0 else q * x**2

    def expectile_loss_grad(x: float) -> float:
        return 2 * (1 - q) * x if x < 0 else 2 * q * x

    def expectile_loss_hessian(x: float) -> float:
        return 2 * (1 - q) if x < 0 else 2 * q

    m = 2 * np.minimum(q, 1 - q)
    M = 2 * np.maximum(q, 1 - q)
    kappa = M / m

    K = bandit.n_arms
    d = bandit.dim_features
    assert T >= K, "Horizon should be higher than number of arms."

    assert bound_theta >= 0, "bound_theta should be positive."
    assert bound_features > 0, "bound_features should be positive."

    visit_times = np.zeros(K)
    mus_hat = np.zeros(K)
    UCB = np.zeros(K)

    theta_hat = np.zeros((d * K))
    theta_bar = np.zeros((d * K))
    # V = np.zeros((d * K, d * K))

    features_history = np.zeros((d * K, T))
    rewards_history = np.zeros(T)

    regrets = np.zeros(T)

    # Observe features
    features = bandit._gen_features()

    # Number of OGD updates
    n = 0

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

        # Update expectile regression
        A = np.zeros((d, K))
        A[:, action] = features[:, action]
        A = A.flatten(order="F")
        features_history[:, t] = A
        rewards_history[t] = reward

        if t == K * warmup:

            def empirical_expectile_loss(theta):
                return (
                    np.sum(
                        [
                            expectile_loss(
                                rewards_history[s] - theta @ features_history[:, s]
                            )
                            for s in range(t + 1)
                        ]
                    )
                    + 0.5 * regularization * (theta**2).sum()
                )

            ret = minimize(
                empirical_expectile_loss,
                theta_hat,
            )

            if ret.success:
                theta_hat_0 = ret.x
                theta_hat = ret.x
            # else:
            # raise Exception("Expectile calculation failed...")
        elif t > K * warmup:
            if t % n_iter_ogd == 0:
                D_loss = [
                    expectile_loss_grad(
                        rewards_history[s] - theta_hat @ features_history[:, s]
                    )
                    for s in range(t - n_iter_ogd + 1, t + 1)
                ]
                grad = np.sum(
                    [
                        D_loss[i] * features_history[:, s] + regularization * theta_hat
                        for i, s in enumerate(range(t - n_iter_ogd + 1, t + 1))
                    ],
                    axis=0,
                )

                # Gradient descent
                theta_hat -= step_size / (n + 1) * np.clip(grad, -grad_clip, grad_clip)

                # Project to keep it close to the initial guess
                norm_diff = np.linalg.norm(theta_hat - theta_hat_0)
                if norm_diff > 2 * bound_theta:
                    theta_hat = (
                        theta_hat_0
                        + (theta_hat - theta_hat_0) / norm_diff * 2 * bound_theta
                    )

            n += 1
            theta_bar = (1 - 1 / n) * theta_bar + 1 / n * theta_hat

        D2_loss = [
            expectile_loss_hessian(
                rewards_history[s] - theta_bar @ features_history[:, s]
            )
            for s in range(t + 1)
        ]

        H0 = np.sum(
            [
                D2_loss[s] * np.outer(features_history[:, s], features_history[:, s])
                for s in range(t + 1)
            ],
            axis=0,
        )

        H = H0 + kappa * regularization * np.eye(d * K)

        H_inv = np.linalg.inv(H)

        # Observe next features
        features = bandit._gen_features()

        # Update exploration bonus
        c = 2 * (
            kappa
            * exploration_bonus_scale
            * np.sqrt(
                2 * np.log(1 / delta)
                - d * K * np.log(kappa * regularization)
                + np.linalg.slogdet(H)[1]
            )
            + np.sqrt(kappa * regularization) * bound_theta
        )

        exploration_bonus = np.zeros(K)
        mus_hat = np.zeros(K)
        for k in bandit.arms:
            A = np.zeros((d, K))
            A[:, k] = features[:, k]
            A = A.flatten(order="F")
            exploration_bonus[k] = c * np.sqrt(A.T @ H_inv @ A)
            mus_hat[k] = theta_bar @ A

        UCB = mus_hat + exploration_bonus

        regrets[t] = bandit.mu_star(features) - bandit.mus(features)[action]

    return regrets
