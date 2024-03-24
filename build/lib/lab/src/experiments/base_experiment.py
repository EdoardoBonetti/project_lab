"""Toolbox for the lab: data loading, plotting, etc."""
from dataclasses import dataclass, field
# import polar
from os import listdir
from pandas import read_csv, concat, DataFrame
import abc
import matplotlib.pyplot as plt


@dataclass
class Single(abc.ABC):
    file_name: str
    file_dir: str
    data: DataFrame = field(init=False)

    def __post_init__(self):
        self.data_reader()

    @abc.abstractmethod
    def data_reader(self):
        """Read the data from the file"""
        pass

    @abc.abstractmethod
    def data_cleaner(self, **kwargs):
        """Clean the data"""
        pass

    @abc.abstractmethod
    def data_saver(self, **kwargs):
        """Save the data"""
        pass

    @abc.abstractmethod
    def data_plotter(self, **kwargs):
        """Plot the data"""
        pass


# @dataclass
# class Multiple(abc.ABC):
#    """Class to represent multiple measurments"""
#
#    dir_name: str
#    data: DataFrame = field(init=False)
#    single_msmt: list = field(init=False)
#
#    def __post_init__(self):
#        self.read_data()
#
#    @abc.abstractmethod
#    def read_data(self):
#        """Read the data from the file"""
#        pass
#
#    @abc.abstractmethod
#    def clean(self, **kwargs):
#        """Clean the data"""
#        pass
#
#    @abc.abstractmethod
#    def plot_mean(self, **kwargs):
#        """Plot the data"""
#        pass
#
#    @abc.abstractmethod
#    def plot_std(self, **kwargs):
#        """Plot the data"""
#        pass
#
#    @abc.abstractmethod
#    def plot_individuals(self, **kwargs):
#        """Plot all the data"""
#        pass
#
#    @abc.abstractmethod
#    def save_plots(self, **kwargs):
#        """Save the plots"""
#        pass
#
#
# @dataclass
# class Group:
#    """Class to represent multiple experiments on different samples"""
#    pass
#
#
#
#
def main():
    """Main implementation to quickly test the code"""
    pass


if __name__ == "__main__":
    main()
