"""define the class SingleDSC that inherits from Single"""


from dataclasses import dataclass, field
# import polar
from os import listdir, makedirs, path
from pandas import read_csv, concat, DataFrame
import abc
from .base_experiment import Single
import matplotlib.pyplot as plt


@dataclass
class SingleDSC(Single):
    """Class to represent a single DSC measurment"""

    def data_reader(self):
        """Read the data from the file"""
        # call the first column Temperature and the second Heat Flow
        self.data = read_csv(self.file_dir + "/" + self.file_name, sep="\t", names=[
                             "Temperature (C)", "Heat Flow (W/g)"])

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

    def data_saver(self, output_dir="output_data", name="output_data.txt"):
        """Save the data"""
        if not path.exists(output_dir):
            makedirs(output_dir)

        self.data.to_csv(f"{output_dir}/{name}", sep="\t")

    def data_plotter(self, **kwargs):
        """Plot the data"""
        plt.plot(self.data["Temperature (C)"], self.data["Heat Flow (W/g)"])
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
