from dataclasses import dataclass, field
from os import listdir
import matplotlib.pyplot as plt
from pandas import DataFrame, read_csv, concat
from sklearn.linear_model import LinearRegression
from scipy.stats import f_oneway


from scipy.signal import savgol_filter
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve

import abc


TEST_MODE = False

# dataclass measurement


@dataclass
class Measurement:
    name: str
    data: DataFrame = field(init=False)

    def __post_init__(self):
        self.data = read_csv(self.name, delimiter='\t', header=0)
        # groub by first column and average the rest
        self.data = self.data.groupby(self.data.columns[0]).mean()
        # get rid of nan values
        self.data = self.data.dropna()

    def clean(self, clean_kwargs):
        self.data = cut(self.data, **clean_kwargs)
        self.data = smooth(self.data, **clean_kwargs)

        self.data = baseline_remove(self.data, **clean_kwargs)
        self.data = smooth(self.data, **clean_kwargs)


def cut(data, **kwargs):
    start = kwargs.get("start", 0)
    end = kwargs.get("end", 100)
    # return the data where index is between start and end
    return data.loc[start:end]


def smooth(data, **kwargs):
    window_length = kwargs.get("window_length", 100)
    polyorder = kwargs.get("polyorder", 3)
    new_data = DataFrame(index=data.index)

    for column in data.columns:

        svflt = savgol_filter(data[column], window_length, polyorder)
        if TEST_MODE:
            new_data[column] = data[column]
            new_data[column+"new"] = svflt

        else:
            new_data[column] = svflt

    return new_data


def baseline_remove(df, **kwargs):
    """Remove baseline"""

    # create new dataframe with same index
    new_df = DataFrame(index=df.index)

    for column in df.columns:

        lam = kwargs.get('lam', 10**13)
        p = kwargs.get('p', 0.99)
        niter = kwargs.get('niter', 100)
        blals = baseline_als(df[column], lam, p, niter)

        if TEST_MODE:
            new_df[column] = df[column]
            new_df[column + "nwe"] = blals
        else:
            new_df[column] = df[column] - blals
        # self.data[column+"nwe"] = baseline_als(self.data[column])

    return new_df


def baseline_als(y, lam, p, niter):
    """Asymmetric Least Squares to find the baseline"""
    L = len(y)
    D = sparse.diags([1, -2, 1], [0, -1, -2], shape=(L, L-2))
    w = np.ones(L)
    for i in range(niter):
        W = sparse.spdiags(w, 0, L, L)
        Z = W + lam * D.dot(D.transpose())
        z = spsolve(Z, w*y)
        w = p * (y > z) + (1-p) * (y < z)
    return z


def cooling():

    clean_kwargs = {  # cut kwargs
        "start": 25,
        "end": 85
    }

    perc2 = []
    perc4 = []
    perc6 = []
    perc8 = []

    for name in listdir("data/cooling/2perc/"):
        meas = Measurement("data/cooling/2perc/"+name)
        meas.clean(clean_kwargs)
        plt.plot(meas.data.index, meas.data, color='C0', linewidth=0.5)
        perc2.append(meas.data)

    for name in listdir("data/cooling/4perc/"):
        meas = Measurement("data/cooling/4perc/"+name)
        meas.clean(clean_kwargs)
        plt.plot(meas.data.index, meas.data, color='C1', linewidth=0.5)
        perc4.append(meas.data)

    for name in listdir("data/cooling/6perc/"):
        meas = Measurement("data/cooling/6perc/"+name)
        meas.clean(clean_kwargs)
        plt.plot(meas.data.index, meas.data, color='C2', linewidth=0.5)
        perc6.append(meas.data)

    for name in listdir("data/cooling/8perc/"):
        meas = Measurement("data/cooling/8perc/"+name)
        meas.clean(clean_kwargs)
        plt.plot(meas.data.index, meas.data, color='C3', linewidth=0.5)
        perc8.append(meas.data)
    # P2 is dataframe containing all the measurements for 2%
    P2 = concat(perc2, axis=1)
    P4 = concat(perc4, axis=1)
    P6 = concat(perc6, axis=1)
    P8 = concat(perc8, axis=1)

    # num rows in P2
    L = min([P2.shape[0], P4.shape[0], P6.shape[0], P8.shape[0]])

    # apply the anova on the first row
    Fvals, pvals = [], []

    for i in range(L):  # L is the length of the shortest dataframe

        Fval, pval = f_oneway(
            P2.iloc[i, :], P4.iloc[i, :], P6.iloc[i, :], P8.iloc[i, :])

        Fvals.append(Fval)
        pvals.append(pval)

    # exct the pvale less than 0.05

    index = P2.index
    plt.plot(index, pvals, color='black', label='p-values')
    # plot 0.05 line
    plt.plot(index, [0.05]*len(index), color='C4',
             label=r'p = 0.05', linewidth=2)
    plt.legend()
    plt.title('p-values for anova test on cooling rso-rbx')
    plt.savefig('anova_cooling.png', dpi=2000)
    # clear the plot
    plt.clf()


def heating():

    clean_kwargs = {  # cut kwargs
        "start": 25,
        "end": 85,
        "p": 0.01,
        "lam": 10**11
    }

    perc2 = []
    perc4 = []
    perc6 = []
    perc8 = []

    for name in listdir("data/heating/2perc/"):
        meas = Measurement("data/heating/2perc/"+name)
        meas.clean(clean_kwargs)
        plt.plot(meas.data.index, meas.data, color='C0', linewidth=0.5)
        perc2.append(meas.data)

    for name in listdir("data/heating/4perc/"):
        meas = Measurement("data/heating/4perc/"+name)
        meas.clean(clean_kwargs)
        plt.plot(meas.data.index, meas.data, color='C1',  linewidth=0.5)
        perc4.append(meas.data)

    for name in listdir("data/heating/6perc/"):
        meas = Measurement("data/heating/6perc/"+name)
        meas.clean(clean_kwargs)
        plt.plot(meas.data.index, meas.data, color='C2',  linewidth=0.5)
        perc6.append(meas.data)

    for name in listdir("data/heating/8perc/"):
        meas = Measurement("data/heating/8perc/"+name)
        meas.clean(clean_kwargs)
        plt.plot(meas.data.index, meas.data, color='C3',  linewidth=0.5)
        perc8.append(meas.data)

    # P2 is dataframe containing all the measurements for 2%
    P2 = concat(perc2, axis=1)
    P4 = concat(perc4, axis=1)
    P6 = concat(perc6, axis=1)
    P8 = concat(perc8, axis=1)

    # num rows in P2
    L = min([P2.shape[0], P4.shape[0], P6.shape[0], P8.shape[0]])

    # apply the anova on the first row
    Fvals, pvals = [], []

    for i in range(L):  # L is the length of the shortest dataframe

        Fval, pval = f_oneway(
            P2.iloc[i, :], P4.iloc[i, :], P6.iloc[i, :], P8.iloc[i, :])

        Fvals.append(Fval)
        pvals.append(pval)

    # exct the pvale less than 0.05

    index = P2.index
    plt.plot(index, pvals, color='black', label='p-values')
    # plot 0.05 line
    plt.plot(index, [0.05]*len(index), color='C4', label=r'p = 0.05')
    plt.legend()
    plt.title('p-values for anova test on heating rso-rbx')
    plt.savefig('anova_heating.png', dpi=2000)
    # clear the plot
    plt.clf()


def main():
    # figure size
    plt.figure(figsize=(10, 5))

    cooling()
    heating()


if __name__ == "__main__":
    main()
