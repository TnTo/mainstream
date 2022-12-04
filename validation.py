# %%
import numpy as np
import matplotlib.pyplot as plt
import pandas
from sklearn.metrics.cluster._supervised import (
    contingency_matrix,
    expected_mutual_information,
    mutual_info_score,
)
from scipy.stats import entropy
import seaborn

# %%
seaborn.set_style("whitegrid")
seaborn.set_context("paper")

# %%
seeds = [1000, 1001, 1002, 1003, 1004]
N = len(seeds)

# %%
mdls = [float(open(f"results_{s}/mdl").read()) for s in seeds]
idx = mdls.index(min(mdls))

# %%
gs = []
lmaxs = []
for i in range(N):
    gs.append([])
    l = 0
    while True:
        try:
            gs[i].append(
                (
                    int(open(f"results_{seeds[i]}/{l}/Bw").read()),
                    int(open(f"results_{seeds[i]}/{l}/Bd").read()),
                )
            )
            l += 1
        except:
            lmaxs.append(l - 1)
            break

# %%
fig, ax = plt.subplots()
for i in range(N):
    c = "red" if i == idx else "blue"
    plt.scatter(*zip(*gs[i]), color=c, marker="D", s=8)

ax.set_xscale("log")
ax.set_yscale("log")

ax.set_xlabel("# Word Groups")
ax.set_ylabel("# Document Groups")

fig.show()

# %%
def adjSoftNMI(m1, m2):
    _, nD = m1.shape
    c = m1 @ m2.T
    norm = np.mean(entropy(c.sum(axis=0)), entropy(c.sum(axis=1)))
    mi = mutual_info_score(None, None, contingency=c)
    emi = expected_mutual_information(c, nD)
    denominator = norm - emi
    if denominator < 0:
        denominator = min(denominator, -np.finfo("float64").eps)
    else:
        denominator = max(denominator, np.finfo("float64").eps)
    ami = (mi - emi) / denominator
    return ami


def SoftNMI(m1, m2):
    c = m1 @ m2.T
    norm = np.mean(entropy(c.sum(axis=0)), entropy(c.sum(axis=1)))
    mi = mutual_info_score(None, None, contingency=c)
    return mi / norm


# %%
np.random.seed(8686)
nmis = []
for l in range(lmaxs[idx] + 1):
    print("L", l)
    bestmd = np.load(f"results_{seeds[idx]}/{l}/Ps.npz")["p_td_d"]
    nDGbest, nD = bestmd.shape
    bestmd = np.argmax(bestmd, axis=0)
    bestmw = np.load(f"results_{seeds[idx]}/{l}/Ps.npz")["p_tw_d"]
    nWGbest, _ = bestmw.shape
    for i in range(N):
        print(i)
        # if i != idx:
        if True:
            for il in range(lmaxs[i] + 1):
                benchmd = np.load(f"results_{seeds[i]}/{il}/Ps.npz")["p_td_d"]
                nDGbench, _ = benchmd.shape
                benchmd = np.argmax(benchmd, axis=0)
                benchrd = np.random.randint(0, nDGbench, (nD,))

                benchmw = np.load(f"results_{seeds[i]}/{il}/Ps.npz")["p_tw_d"]
                nWGbench, _ = benchmw.shape
                benchrw = np.random.rand(nWGbench, nD)
                benchrw = benchrw / benchrw.sum(axis=0)

                # https://github.com/scikit-learn/scikit-learn/blob/f3f51f9b6/sklearn/metrics/cluster/_supervised.py

                cDm = contingency_matrix(bestmd, benchmd)
                cDr = contingency_matrix(bestmd, benchrd)
                cWm = bestmw @ benchmw.T
                cWr = bestmw @ benchrw.T

                miDm = mutual_info_score(None, None, contingency=cDm)
                miDr = mutual_info_score(None, None, contingency=cDr)
                miWm = mutual_info_score(None, None, contingency=cWm)
                miWr = mutual_info_score(None, None, contingency=cWr)

                nDm = np.mean([entropy(bestmd), entropy(benchmd)])
                nDr = np.mean([entropy(bestmd), entropy(benchrd)])
                nWm = np.mean([entropy(cWm.sum(axis=0)), entropy(cWm.sum(axis=1))])
                nWr = np.mean([entropy(cWr.sum(axis=0)), entropy(cWr.sum(axis=1))])

                eDm = expected_mutual_information(cDm, nD)
                eDr = expected_mutual_information(cDr, nD)
                eWm = expected_mutual_information(cWm, nD)
                eWr = expected_mutual_information(cWr, nD)

                nmis.append(
                    (
                        l,
                        i,
                        il,
                        nDGbest,
                        nWGbest,
                        nDGbench,
                        nWGbench,
                        miDm,
                        miDr,
                        miWm,
                        miWr,
                        nDm,
                        nDr,
                        nWm,
                        nWr,
                        eDm,
                        eDr,
                        eWm,
                        eWr,
                    )
                )
df = pandas.DataFrame(
    nmis,
    columns=[
        "BestLevel",
        "Benchmark",
        "BenchmarkLevel",
        "BestNDGroups",
        "BestNWGroups",
        "BenchmarkNDGroups",
        "BenchmarkNWGroups",
        "MIDModel",
        "MIDRandom",
        "MIWModel",
        "MIWRandom",
        "NormDModel",
        "NormDRandom",
        "NormWModel",
        "NormWRandom",
        "EDModel",
        "EDRandom",
        "EWModel",
        "EWRandom",
    ],
)
df.to_csv("nmis.csv")

# %%
df = pandas.read_csv("nmis.csv", index_col=0)

for T in ["D", "W"]:
    for m in ["Model", "Random"]:
        df[f"NMI{T}{m}"] = df[f"MI{T}{m}"] / df[f"Norm{T}{m}"]
        df[f"AMI{T}{m}"] = (df[f"MI{T}{m}"] - df[f"E{T}{m}"]) / (
            df[f"Norm{T}{m}"] - df[f"E{T}{m}"]
        )
    df[f"dMI{T}"] = (df[f"MI{T}Model"] - df[f"MI{T}Random"]) / df[f"MI{T}Model"]
    df[f"dNMI{T}"] = (df[f"NMI{T}Model"] - df[f"NMI{T}Random"]) / df[f"NMI{T}Model"]
    df[f"dAMI{T}"] = (df[f"AMI{T}Model"] - df[f"AMI{T}Random"])

df = df.query(
    "BestNDGroups > 1 and BestNWGroups > 1 and BenchmarkNDGroups > 1 and BenchmarkNWGroups > 1"
)

# %%
for T in ["D", "W"]:
    g = seaborn.FacetGrid(
        df.melt(
            id_vars=[
                "BestLevel",
                "Benchmark",
                "BenchmarkLevel",
                f"BestN{T}Groups",
                f"BenchmarkN{T}Groups",
            ],
            value_vars=[
                f"MI{T}Model",
                f"NMI{T}Model",
                f"AMI{T}Model",
                f"dMI{T}",
                f"dNMI{T}",
                f"dAMI{T}",
            ],
        ),
        hue="Benchmark",
        col=f"BestN{T}Groups",
        row="variable",
        sharey="row",
        aspect=1.5,
    )
    g.map(seaborn.scatterplot, f"BenchmarkN{T}Groups", "value", markers="D")
    for ax in g.axes.ravel():
        ax.set_xscale("log")
    plt.show()

# %%
