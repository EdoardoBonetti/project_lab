"""Toolbox for the lab: data loading, plotting, etc."""
from dataclasses import dataclass, field
# import polar
from os import listdir
from pandas import read_csv, concat, DataFrame
import abc
import matplotlib.pyplot as plt
import numpy as np


@dataclass
class Stats:
    


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
    def data_check(self, **kwargs):
        """Save the data"""
        pass

    @abc.abstractmethod
    def data_saver(self, **kwargs):
        """Save the data"""
        pass

    @abc.abstractmethod
    def data_plotter(self, **kwargs):
        """Plot the data"""
        pass


@dataclass
class Repeated(abc.ABC):
    """Class to represent repeated measurments"""
    file_dir: str
    data: DataFrame = field(init=False)
    single_mesurement: list = field(init=False)
    stats: Stats() = field(init=False)

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
    def data_check(self, **kwargs):
        """Save the data"""
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
