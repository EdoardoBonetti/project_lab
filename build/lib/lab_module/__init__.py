"""Toolbox for the lab: data loading, plotting, etc."""
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

# define an abstracr class Measurment and then define a subclass for each type of data TPA, RSO, etc.


# Global variables
# SHOW_PLOTS = True
TESTING_MODE = True


@dataclass
class Measurment(abc.ABC):
    filename: str
    data: DataFrame = field(init=False)

    def __post_init__(self):
        self.read_data()

    @abc.abstractmethod
    def read_data(self):
        """Read the data from the file"""
        pass

    @abc.abstractmethod
    def clean(self, **kwargs):
        """Clean the data"""
        pass

    @abc.abstractmethod
    def plot(self, **kwargs):
        """Plot the data"""
        pass


@dataclass
class Multiple(abc.ABC):
    """Class to represent multiple measurments"""

    dir_name: str
    data: DataFrame = field(init=False)
    single_msmt: list = field(init=False)

    def __post_init__(self):
        self.read_data()

    @abc.abstractmethod
    def read_data(self):
        """Read the data from the file"""
        pass

    @abc.abstractmethod
    def clean(self, **kwargs):
        """Clean the data"""
        pass

    @abc.abstractmethod
    def plot_mean(self, **kwargs):
        """Plot the data"""
        pass

    @abc.abstractmethod
    def plot_std(self, **kwargs):
        """Plot the data"""
        pass

    @abc.abstractmethod
    def plot_individuals(self, **kwargs):
        """Plot all the data"""
        pass

    @abc.abstractmethod
    def save_plots(self, **kwargs):
        """Save the plots"""
        pass


# demo.py
@dataclass
class DSC(Measurment):
    """Class to represent a measurment for DSC"""

    def read_data(self):
        """Read the data from the file"""
        self.data = read_csv(self.filename, sep='\t', header=None)
        # self.data = self.data.drop(self.data.columns[0], axis=1)
        # self.data = self.data.apply(lambda x: x.str.replace(',', '.'))
        self.data = self.data.astype(float)
        self.data.columns = ['Temperature', self.filename]
        # self.data['Displacement'] = self.data['Displacement'].apply(
        #    lambda x: round(x, 1))

    def clean(self, **kwargs):
        """
        Clean the data:
        - remove all rows with NaN
        - remove all rows with temperature outside the range [temp_min, temp_max]
        - shift the data to the right
        Remove all rows with NaN"""
        self.data = self.data.astype(float)
        temp_min = kwargs.get('temp_min', 0)
        temp_max = kwargs.get('temp_max', 100)
        self.data = self.data[self.data["Temperature"] > temp_min]
        self.data = self.data[self.data["Temperature"] < temp_max]
        self.data = self.data.dropna()
        self.data = self.data.groupby(by='Temperature').mean()
        self.data = smooth(self.data, **kwargs)
        self.data = baseline_remove(self.data, **kwargs)

    def plot(self, **kwargs):
        """Plot the data"""
        plt.plot(self.data)  # , label=self.filename)
        if not kwargs.get("overlap_plot", False):
            plt.show()


# derive a class for multiple DSC measurments
@dataclass
class MultipleDSC(Multiple):
    """Class to represent multiple DSC measurments"""

    def read_data(self):
        """Read the data from the file"""
        files = listdir(self.dir_name)
        # create a list of single measurments
        self.single_msmt = []
        # read the data from each file
        for file in files:
            self.single_msmt.append(DSC(self.dir_name + '/' + file))
        # concatenate all the data using the temperature as index
        self.data = []

    def clean(self, **kwargs):
        """Clean the data"""
        for msmt in self.single_msmt:
            msmt.clean(**kwargs)
        # concatenate all the data using the temperature as index
        self.data = concat([msmt.data for msmt in self.single_msmt],
                           axis=1)  # , join='inner')
        # remove all rows with NaN
        self.data = self.data.dropna()

    def plot_mean(self, **kwargs):
        """Plot the data"""

        plt.plot(self.data.mean(axis=1), **kwargs)

    def plot_std(self, **kwargs):
        """Plot the data"""
        plt.fill_between(self.data.index, self.data.mean(axis=1) - self.data.std(axis=1),
                         self.data.mean(axis=1) + self.data.std(axis=1), alpha=0.5, **kwargs)

    def plot_individuals(self, **kwargs):
        """Plot all the data"""

        for msmt in self.single_msmt:
            msmt.plot(overlap_plot=True, **kwargs)

    def save_plots(self, **kwargs):
        """Save the plots"""
        save_name = kwargs.get("save_name", kwargs.get("title", "DSC thermogram").replace(
            "/", "").replace("%", ""))
        plt.savefig(save_name + '.png')

    def prepare_plot(self, **kwargs):
        """Prepare the plot"""
        plt.figure(figsize=(10, 5))
        plt.xlabel("Temperature [°C]")
        plt.ylabel("Heat flow [mW]")
        plt.grid()
        plt.title(kwargs.get("title", "DSC thermogram"))
        plt.tight_layout()


def baseline_remove(df, **kwargs):
    """Remove baseline"""
    print(len(df))
    # apply baseline to every column
    for column in df.columns:
        if kwargs.get('draw_baseline', False):
            df[column +
               "nwe"] = baseline_als(df[column], **kwargs)
        else:
            df[column] = df[column] - \
                baseline_als(df[column], **kwargs)
        # self.data[column+"nwe"] = baseline_als(self.data[column])

    return df


def baseline_als(y, lam, p, niter, **kwargs):
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


def smooth(df, **kwargs):
    # we need to smooth eace column of the dataframe usifng the Savitzky-Golay filter

    # take the min in index and max in index
    min_index = df.index.min()
    max_index = df.index.max()
    # create a new index
    new_index = np.arange(round(min_index, 1), round(max_index, 1), 0.1)
    # create a new dataframe
    new_df = DataFrame(index=new_index)
    # interpolate the data
    for column in df.columns:
        new_df[column] = np.interp(new_index, df.index, df[column])
        new_df[column] = savgol_filter(new_df[column], 51, 3)
    return new_df

# the anova function takes as args multiples and returns the p-value of the anova test


def anova(all_data):
    # all data contains a list of dataframes, we need to comapare row by row if the index is the same
    # if not we need to interpolate the data



def test_dsc():
    # test single file

    dsc = DSC(
        "DSC/RBXRSO/cooling/2perc/231116_RBX_RSO_2perc_B_2st_cooling_raw_data.txt")
    print(dsc.data)
    # dsc.plot()
    kwargs = {'temp_min': 0, 'temp_max': 85,
              "p": 0.99, "lam": 10**(5), "niter": 100}
    # add to the kwargs draw_baseline = True to draw the baseline
    kwargs['draw_baseline'] = False
    dsc.clean(**kwargs)
    print(dsc.data)
    # dsc.plot()


def test_multiple_dsc_cooling():
    kwargs = {'temp_min': 0, 'temp_max': 85,
              "p": 0.99, "lam": 10**(5), "niter": 100}

    # test multiple files
    multiple_dsc = MultipleDSC("DSC/LARD/cooling/lard")
    # print(multiple_dsc.data)
    multiple_dsc.clean(**kwargs)
    print(multiple_dsc.data)

    kwargs_plot = {"title": "DSC thermogram Pork"}

    multiple_dsc.prepare_plot(**kwargs_plot)
    # plot the data
    multiple_dsc.plot_mean(label="mean")
    multiple_dsc.plot_std(label="std")
    multiple_dsc.plot_individuals()

    plt.legend()
    plt.show()


def test_multiple_dsc_rbxreo_cooling():
    kwargs = {'temp_min': 10, 'temp_max': 85,
              "p": 0.99, "lam": 10**(12), "niter": 200}

    # test multiple files
    multiple_2 = MultipleDSC("DSC/RBXRSO/cooling/2perc")
    multiple_4 = MultipleDSC("DSC/RBXRSO/cooling/4perc")
    multiple_6 = MultipleDSC("DSC/RBXRSO/cooling/6perc")
    multiple_8 = MultipleDSC("DSC/RBXRSO/cooling/8perc")
    # print(multiple_dsc.data)
    multiple = [multiple_2, multiple_4, multiple_6, multiple_8]

    all_data = []
    for msmt in multiple:
        print(msmt.data)
        msmt.clean(**kwargs)
        print(msmt.data)
        all_data.append(msmt.data)

        kwargs_plot = {"title": "DSC thermogram Pork"}

        msmt.prepare_plot(**kwargs_plot)
        # msmt.plot_mean(label="mean")
        # msmt.plot_std(label="std")
        msmt.plot_individuals()
        plt.legend()
        plt.show()

    anova(all_data)

    # multiple_dsc.clean(**kwargs)
    # print(multiple_dsc.data)

#    kwargs_plot = {"title": "DSC thermogram Pork"}
#
#    multiple_2.prepare_plot(**kwargs_plot)
#    # plot the data
#    multiple_2.plot_mean(label="mean")
#    multiple_2.plot_std(label="std")
#    multiple_2.plot_individuals()
#
#    plt.legend()
#    plt.show()


def main():
    """Main implementation to quickly test the code"""

    # test_dsc()
    test_multiple_dsc_rbxreo_cooling()


if __name__ == "__main__":
    main()