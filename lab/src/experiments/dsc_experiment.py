"""define the class SingleDSC that inherits from Single"""


from dataclasses import dataclass, field
# import polar
from os import listdir, makedirs, path
from pandas import read_csv, concat, DataFrame
import abc
from .base_experiment import Single
from ..utils import baseline_als
import matplotlib.pyplot as plt


@dataclass
class SingleDSC(Single):
    """Class to represent a single DSC measurment"""

    def data_reader(self):
        """Read the data from the file"""
        # call the first column Temperature and the second Heat Flow
        self.data = read_csv(self.file_dir + "/" + self.file_name, sep="\t", names=[
                             "Temperature (C)", "Heat Flow (W/g)"])

    def check(self, **kwargs):
        # create a dataframe with temperature, old heat flow and new heat flow
        self.old_data = self.data
        self.old_data["baseline"] = baseline_als(
            self.old_data["Heat Flow (W/g)"], **kwargs)

        plt.plot(self.old_data["Temperature (C)"],
                 self.data["Heat Flow (W/g)"])
        plt.plot(self.old_data["Temperature (C)"], self.old_data["baseline"])


    def data_cleaner(self, **kwargs):
        """Clean the data"""
        self.data = self.data.dropna()
        # check if min_temp and max_temp are in kwargs
        if "min_temp" in kwargs:
            min_temp = kwargs["min_temp"]
            self.data = self.data[self.data["Temperature (C)"] > min_temp]

        if "max_temp" in kwargs:
            max_temp = kwargs["max_temp"]
            self.data = self.data[self.data["Temperature (C)"] < max_temp]

        self.data = self.data.groupby("Temperature (C)").mean().reset_index()
        self.baseline = baseline_als(
            self.data["Heat Flow (W/g)"], **kwargs)

        self.data["Heat Flow (W/g)"] -= self.baseline

    def data_saver(self, output_dir="output_data", name="output_data.txt"):
        """Save the data"""
        if not path.exists(output_dir):
            makedirs(output_dir)

        self.data.to_csv(f"{output_dir}/{name}", sep="\t")

    def data_plotter(self, **kwargs):
        """Plot the data"""
        plt.plot(self.data["Temperature (C)"], self.data["Heat Flow (W/g)"])
#        plt.plot(self.data["Temperature (C)"],
#                 self.data["Heat Flow BaseLine (W/g) "])

        for column in self.data.columns:
            if "BaseLine" in column:
                plt.plot(self.data["Temperature (C)"],
                         self.data[column], label=column)
        plt.xlabel("Temperature (°C)")
        plt.ylabel("Heat Flow (W/g)")
        plt.title("DSC Measurement")
        plt.show()





def main():
    """Main implementation to quickly test the code"""
    dsc = SingleDSC("data/DSC_1.txt")
    dsc.data_reader()
    dsc.data_cleaner()
    dsc.data_plotter()


if __name__ == "__main__":
    main()
