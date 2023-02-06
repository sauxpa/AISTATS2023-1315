import argparse
import pickle
from joblib import Parallel, delayed
import multiprocessing as mp
from time import time
import os
import numpy as np
from scipy.stats import rv_continuous, rv_discrete
from tqdm import tqdm

from bandits_lib_lite import (
    LinearBanditEnv,
    EntropicRiskUCB,
    LinUCB,
    EntropicRiskOGDUCB,
    get_bernoulli_entropic_risk,
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


class ConstantContexts(rv_continuous):
    """
    Generates constant contexts of dim (d, K).
    Similar to the standard multi-armed bandit problem.
    """

    def __init__(self, X):
        super(ConstantContexts, self).__init__()
        self.X = X

    def rvs(self):
        return self.X


class Bernoulli:
    """
    Generates constant contexts of dim (d, K).
    Similar to the standard multi-armed bandit problem.
    """

    def __init__(self, xk, pk):
        assert len(xk) == len(pk)
        self.ik = np.arange(len(xk))
        self.support = np.array(xk)
        values = (self.ik, pk)
        self.gen = rv_discrete(values=values)

    def rvs(self, size=1):
        return self.support[self.gen.rvs(size=size)]


def run(T, M, reg, eb_scale, n_iter_ogd, step_size, warmup, verbose):
    """
    A single run of multiple competing bandit algorithms.
    """
    # Init
    gamma = 1.0

    K = 2

    theta = np.array(
        [
            [get_bernoulli_entropic_risk(gamma, 0.5, 1, -1), 0.0],
            [0.0, get_bernoulli_entropic_risk(gamma, 0.25, 2, -2)],
        ]
    )

    loc = np.array(
        [
            [1, 0],
            [0, 1],
        ]
    )
    features_dist = ConstantContexts(loc)

    noise_dist = [
        Bernoulli(
            [
                1.0 - get_bernoulli_entropic_risk(gamma, 0.5, 1, -1),
                -1.0 - get_bernoulli_entropic_risk(gamma, 0.5, 1, -1),
            ],
            [0.5, 0.5],
        ),
        Bernoulli(
            [
                2.0 - get_bernoulli_entropic_risk(gamma, 0.25, 2, -2),
                -2.0 - get_bernoulli_entropic_risk(gamma, 0.25, 2, -2),
            ],
            [0.25, 0.75],
        ),
    ]

    bandit = LinearBanditEnv(
        theta=theta,
        features_dist=features_dist,
        noise_dist=noise_dist,
    )

    delta = 0.05
    # delta = 1 / T

    regrets_ucb = np.zeros((T, M)) * np.nan
    regrets_entropic_ucb = np.zeros((T, M)) * np.nan
    regrets_entropic_ogducb = np.zeros((T, M)) * np.nan

    time_ucb = np.zeros(M) * np.nan
    time_entropic_ucb = np.zeros(M) * np.nan
    time_entropic_ogducb = np.zeros(M) * np.nan

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
        regrets_entropic_ucb[:, m] = EntropicRiskUCB(
            bandit,
            T,
            gamma,
            delta,
            reg,
            exploration_bonus_scale=eb_scale,
            support_hi=1.0,
            support_lo=-1.0,
            bound_theta=1.0,
            bound_features=1.0,
            seed=seed,
            warmup=warmup,
            verbose=verbose,
        )
        end = time()
        time_entropic_ucb[m] = end - start

        start = time()
        regrets_entropic_ogducb[:, m] = EntropicRiskOGDUCB(
            bandit,
            T,
            gamma,
            delta,
            reg,
            exploration_bonus_scale=eb_scale,
            step_size=step_size,
            n_iter_ogd=n_iter_ogd,
            support_hi=1.0,
            support_lo=-1.0,
            bound_theta=1.0,
            bound_features=1.0,
            seed=seed,
            warmup=warmup,
            verbose=verbose,
        )
        end = time()
        time_entropic_ogducb[m] = end - start

    return {
        "ucb": regrets_ucb,
        "entropic_ucb": regrets_entropic_ucb,
        "entropic_ogducb": regrets_entropic_ogducb,
        "time_ucb": time_ucb,
        "time_entropic_ucb": time_entropic_ucb,
        "time_entropic_ogducb": time_entropic_ogducb,
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
    T, M, reg, eb_scale, n_iter_ogd, step_size, warmup, verbose = args
    new_args = (T, M // cpu + 1, reg, eb_scale, n_iter_ogd, step_size, warmup, verbose)
    if parallel:
        res_ = Parallel(n_jobs=cpu)(delayed(MC_xp)(new_args) for _ in range(cpu))
        res = {}
        res["ucb"] = np.concatenate([res_[i]["ucb"] for i in range(cpu)], axis=1)
        res["entropic_ucb"] = np.concatenate(
            [res_[i]["entropic_ucb"] for i in range(cpu)], axis=1
        )
        res["entropic_ogducb"] = np.concatenate(
            [res_[i]["entropic_ogducb"] for i in range(cpu)], axis=1
        )
        res["time_ucb"] = np.concatenate([res_[i]["time_ucb"] for i in range(cpu)])
        res["time_entropic_ucb"] = np.concatenate(
            [res_[i]["time_entropic_ucb"] for i in range(cpu)]
        )
        res["time_entropic_ogducb"] = np.concatenate(
            [res_[i]["time_entropic_ogducb"] for i in range(cpu)]
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


cap = "entropic_ucb_" + str(int(np.random.uniform() * 1e6))
print(cap)
res, traj = multiprocess_MC(
    (T, M, reg, eb_scale, n_iter_ogd, step_size, warmup, verbose),
    pickle_path=pickle_path,
    caption=cap,
    parallel=parallel,
)
print(cap)
