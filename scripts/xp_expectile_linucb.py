import argparse
import pickle
from joblib import Parallel, delayed
import multiprocessing as mp
from time import time
import os
import numpy as np
from scipy.stats import norm, rv_continuous
from tqdm import tqdm
from typing import List

from bandits_lib_lite import (
    LinearBanditEnv,
    ExpectileUCB,
    LinUCB,
    ExpectileOGDUCB,
    AsymmetricNorm,
)


# Get arguments
parser = argparse.ArgumentParser()
parser.add_argument("-T", default=5000, type=int)
parser.add_argument("-M", default=10, type=int)
parser.add_argument("--reg", default=1.0, type=float)
parser.add_argument("--eb_scale", default=1.0, type=float)
parser.add_argument("--n_iter_ogd", default=10, type=int)
parser.add_argument("--step_size", default=0.1, type=float)
parser.add_argument("--warmup", default=1, type=int)
parser.add_argument("-v", default=1, type=int)
parser.add_argument("--path", "-p", default="results", type=str)
parser.add_argument("--parallel", default=1, type=int)

# Set arguments
args = parser.parse_args()
T = args.T
M = args.M
reg = args.reg
eb_scale = args.eb_scale
n_iter_ogd = args.n_iter_ogd
step_size = args.step_size
warmup = args.warmup
verbose = args.v > 0
pickle_path = args.path
parallel = args.parallel > 0


class NormalizedGaussian(rv_continuous):
    """
    Generates K random bounded features in R^d using the following rule:
        1) draw K d-dimensional Gaussian vectors (independent components),
        2) normalize each vector by its L2 norm,
        3) stack the K vectors and return the resulting matrix.
    """

    def __init__(
        self,
        K=1,
        loc: List[float] = [0.0],
        scale: List[float] = [1.0],
    ):
        super(NormalizedGaussian, self).__init__()
        self.K = K
        self.loc = loc
        self.scale = scale
        self.d = len(loc)

    def rvs(self):
        X = [
            norm.rvs(loc=self.loc[:, k], scale=self.scale[k]).reshape(-1, 1)
            for k in range(self.K)
        ]
        X = [x / np.linalg.norm(x) for x in X]
        return np.concatenate(X, axis=1)


def run(T, M, reg, eb_scale, n_iter_ogd, step_size, warmup, verbose):
    """
    A single run of multiple competing bandit algorithms.
    """
    # Init
    q = 0.1

    K = 2
    d = 3

    theta = np.array(
        [
            [0.9, 0.9],
            [0.0, 0.0],
            [0.1, 0.1],
        ]
    )

    theta /= np.linalg.norm(theta, axis=0)

    loc = np.array(
        [
            [1, 0],
            [0, 1],
            [0, 0],
        ]
    )
    scale = np.array([0.1, 0.1])
    features_dist = NormalizedGaussian(K=K, loc=loc, scale=scale)

    noise_dist = [
        AsymmetricNorm(q, loc=0.0, scale=0.5, a=-20, b=20),
        AsymmetricNorm(q, loc=0.0, scale=1.5, a=-20, b=20),
    ]

    bandit = LinearBanditEnv(
        theta=theta,
        features_dist=features_dist,
        noise_dist=noise_dist,
    )

    delta = 0.05
    # delta = 1 / T

    regrets_ucb = np.zeros((T, M)) * np.nan
    regrets_expectile_ucb = np.zeros((T, M)) * np.nan
    regrets_expectile_ogducb = np.zeros((T, M)) * np.nan

    time_ucb = np.zeros(M) * np.nan
    time_expectile_ucb = np.zeros(M) * np.nan
    time_expectile_ogducb = np.zeros(M) * np.nan

    for m in tqdm(range(M)):
        seed = int(np.random.uniform() * 1e6)

        start = time()
        regrets_ucb[:, m] = LinUCB(
            bandit,
            T,
            delta,
            reg,
            exploration_bonus_scale=eb_scale,
            bound_theta=1.0,
            bound_features=1.0,
            seed=seed,
            warmup=warmup,
            verbose=verbose,
        )
        end = time()
        time_ucb[m] = end - start

        start = time()
        regrets_expectile_ucb[:, m] = ExpectileUCB(
            bandit,
            T,
            q,
            delta,
            reg,
            exploration_bonus_scale=eb_scale,
            bound_theta=1.0,
            bound_features=1.0,
            seed=seed,
            warmup=warmup,
            verbose=verbose,
        )
        end = time()
        time_expectile_ucb[m] = end - start

        start = time()
        regrets_expectile_ogducb[:, m] = ExpectileOGDUCB(
            bandit,
            T,
            q,
            delta,
            reg,
            exploration_bonus_scale=eb_scale,
            step_size=step_size,
            n_iter_ogd=n_iter_ogd,
            bound_theta=1.0,
            bound_features=1.0,
            seed=seed,
            warmup=warmup,
            verbose=verbose,
        )
        end = time()
        time_expectile_ogducb[m] = end - start

    return {
        "ucb": regrets_ucb,
        "expectile_ucb": regrets_expectile_ucb,
        "expectile_ogducb": regrets_expectile_ogducb,
        "time_ucb": time_ucb,
        "time_expectile_ucb": time_expectile_ucb,
        "time_expectile_ogducb": time_expectile_ogducb,
    }


def MC_xp(args, pickle_path=None, caption="xp"):
    (
        T,
        M,
        reg,
        eb_scale,
        n_iter_ogd,
        step_size,
        warmup,
        verbose,
    ) = args
    res = run(T, M, reg, eb_scale, n_iter_ogd, step_size, warmup, verbose)

    if pickle_path is not None:
        pickle.dump(res, open(os.path.join(pickle_path, caption + ".pkl"), "wb"))
    return res


def multiprocess_MC(args, pickle_path=None, caption="xp", parallel=True):
    t0 = time()
    cpu = mp.cpu_count()
    print("Running on %i clusters" % cpu)
    T, M, reg, scale, n_iter_ogd, step_size, warmup, verbose = args
    new_args = (T, M // cpu + 1, reg, eb_scale, n_iter_ogd, step_size, warmup, verbose)
    if parallel:
        res_ = Parallel(n_jobs=cpu)(delayed(MC_xp)(new_args) for _ in range(cpu))
        res = {}
        res["ucb"] = np.concatenate([res_[i]["ucb"] for i in range(cpu)], axis=1)
        res["expectile_ucb"] = np.concatenate(
            [res_[i]["expectile_ucb"] for i in range(cpu)], axis=1
        )
        res["expectile_ogducb"] = np.concatenate(
            [res_[i]["expectile_ogducb"] for i in range(cpu)], axis=1
        )
        res["time_ucb"] = np.concatenate([res_[i]["time_ucb"] for i in range(cpu)])
        res["time_expectile_ucb"] = np.concatenate(
            [res_[i]["time_expectile_ucb"] for i in range(cpu)]
        )
        res["time_expectile_ogducb"] = np.concatenate(
            [res_[i]["time_expectile_ogducb"] for i in range(cpu)]
        )
    else:
        res = MC_xp(args)

    info = {
        "T": T,
        "M": M,
        "regularization": reg,
        "exploration_bonus_scale": eb_scale,
        "n_iter_ogd": n_iter_ogd,
        "step_size": step_size,
        "warmup": warmup,
    }
    xp_container = {"results": res, "info": info}

    if pickle_path is not None:
        pickle.dump(
            xp_container, open(os.path.join(pickle_path, caption + ".pkl"), "wb")
        )
    print("Execution time: {:.0f} seconds".format(time() - t0))
    return xp_container


cap = "expectile_linucb_" + str(int(np.random.uniform() * 1e6))
print(cap)
res, traj = multiprocess_MC(
    (T, M, reg, eb_scale, n_iter_ogd, step_size, warmup, verbose),
    pickle_path=pickle_path,
    caption=cap,
    parallel=parallel,
)
print(cap)
