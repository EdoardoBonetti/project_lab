
import matplotlib.pyplot as plt

# import slider from matplotlib
from matplotlib.widgets import Slider

from plotly.subplots import make_subplots
import numpy as np
import plotly.graph_objs as go


import pandas as pd
from ipywidgets import interact, FloatRangeSlider, IntRangeSlider
from ipywidgets import FloatSlider, fixed, BoundedIntText, BoundedFloatText
from scipy.signal import savgol_filter
from scipy import sparse
from scipy.sparse.linalg import spsolve
from os import listdir


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


def interactive_plot(dfs, cut, filter, lam, p, niter):

    # fig = go.Figure().set_subplots(rows=1, cols=2, subplot_titles=("Heat Flux", "filtered Heat Flux"))

    # create matplotlib subplots
    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.suptitle('A tale of 2 subplots')

    # from name read the file
    for df in dfs:

        begin = cut[0]
        end = cut[1]

        df = df[df['Temperature'] > begin]
        df = df[df['Temperature'] < end]

        polyorder = filter[0]
        window_length = filter[1]

        svflt = savgol_filter(df["Heat Flux"], window_length, polyorder)
        baseline = baseline_als(df["Heat Flux"], lam, p, niter)

        ax1.plot(df['Temperature'], df['Heat Flux'], color='C0', linewidth=0.5)
        ax1.plot(df['Temperature'], baseline, color='C1', linewidth=0.5)
        ax2.plot(df['Temperature'], svflt-baseline, color='C0', linewidth=0.5)

    # add slider for a vertical line on the temperature

    plt.show()
    # clear the plot


cut = FloatRangeSlider(
    value=[25, 85],
    min=0,
    max=100.0,
    step=0.1,
    description='Range of Temperature:',
    disabled=False,
    continuous_update=False,
    orientation='horizontal',
    readout=True,
    readout_format='.1f',
)

filer = IntRangeSlider(
    value=[1, 50],
    min=1,
    max=100,
    step=1,
    description='poly-wind for filter',
    disabled=False,
    continuous_update=False,
    orientation='horizontal',
    readout=True,
    readout_format='d')

# lam , p , niter are values not sliders

lam = BoundedFloatText(
    value=10**8,
    min=10,
    max=10**13,
    step=10**8,
    description='Lambda:',
    disabled=False
)

p = BoundedFloatText(
    value=0.99,
    min=0.01,
    max=0.99,
    step=0.01,
    description='p:',
    disabled=False
)

niter = BoundedIntText(
    value=100,
    min=1,
    max=1000,
    step=1,
    description='niter:',
    disabled=False
)


location = "data/cooling/2perc"

dfs = []

for file in listdir(location):
    df = pd.read_csv(location + "/" + file, sep='\t',
                     header=None, names=['Temperature', 'Heat Flux'])
    df = df.groupby('Temperature').mean().reset_index()
    dfs.append(df)


interact(interactive_plot, dfs=fixed(dfs), cut=cut,
         filter=filer, lam=lam, p=p, niter=niter)
