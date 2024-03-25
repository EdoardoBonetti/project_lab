"""define the class SingleDSC that inherits from Single"""


from dataclasses import dataclass, field
# import polar
from os import listdir, makedirs, path
from pandas import read_csv, concat, DataFrame
import abc
from .base_experiment import Single, Repeated
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

    def data_check(self, **kwargs):
        # this function check that we used for cleaning are ok

        #  to do so I create a deepcopy of the dataframe with 3 columns, the temperature, the heat flow and the baseline
        # drop the NaN values

        check_df = self.data.copy()

        check_df = check_df.dropna()
        # group the data by temperature
        check_df = check_df.groupby("Temperature (C)").mean().reset_index()

        # if min_temp and max_temp are in kwargs
        if "min_temp" in kwargs:
            min_temp = kwargs["min_temp"]
            check_df = check_df[check_df["Temperature (C)"] > min_temp]

        if "max_temp" in kwargs:
            max_temp = kwargs["max_temp"]
            check_df = check_df[check_df["Temperature (C)"] < max_temp]

        check_df["Heat Flow BaseLine (W/g)"] = baseline_als(
            check_df["Heat Flow (W/g)"], **kwargs)

        # create 2 plots side by side
        fig, axs = plt.subplots(1, 2, figsize=(10, 5))

        # plot the heat flow
        axs[0].plot(check_df["Temperature (C)"], check_df["Heat Flow (W/g)"])
        axs[0].plot(check_df["Temperature (C)"],
                    check_df["Heat Flow BaseLine (W/g)"])

        axs[0].set_title("Heat Flow - Baseline")

        # plot the baseline
        axs[1].plot(check_df["Temperature (C)"],
                    check_df["Heat Flow (W/g)"] - check_df["Heat Flow BaseLine (W/g)"])
        axs[1].set_title("Subtraction")

        # set a title for the whole figure
        p = kwargs["p"]
        lam = kwargs["lam"]
        niter = kwargs["niter"]

        # lambda needs to be in scientific notation
        fig.suptitle("Check the cleaning process: p = " + str(p) +
                     ", lam = " + "{:.3e}".format(lam) + ", niter = " + str(niter))

    def data_cleaner(self, **kwargs):
        """Clean the data"""

        # drop the NaN values
        self.data = self.data.dropna()

        # group by mean temperature
        self.data = self.data.groupby("Temperature (C)").mean().reset_index()

        # if min_temp and max_temp are in kwargs
        if "min_temp" in kwargs:
            min_temp = kwargs["min_temp"]
            self.data = self.data[self.data["Temperature (C)"] > min_temp]

        if "max_temp" in kwargs:
            max_temp = kwargs["max_temp"]
            self.data = self.data[self.data["Temperature (C)"] < max_temp]

        self.data["Heat Flow (W/g)"] = self.data["Heat Flow (W/g)"] - \
            baseline_als(
            self.data["Heat Flow (W/g)"], **kwargs)

    def data_saver(self, output_dir="output_data", name="output_data.txt"):
        """Save the data"""
        if not path.exists(output_dir):
            makedirs(output_dir)

        self.data.to_csv(f"{output_dir}/{name}", sep="\t")

    def data_plotter(self, **kwargs):
        """Plot the data"""
        plt.plot(self.data["Temperature (C)"], self.data["Heat Flow (W/g)"])
        plt.plot()
        plt.xlabel("Temperature (°C)")
        plt.ylabel("Heat Flow (W/g)")
        plt.title("DSC Measurement")
        if "save" in kwargs:
            plt.savefig(kwargs["output_dir"] + "/" + kwargs["name"])
        plt.show()


class RepeatedDSC(Repeated):
    """Class to represent repeated DSC measurments"""

    def data_reader(self):
        """Read the data from the file"""
        # list all the files in the directory
        files = listdir(self.file_dir)

        # create a list of SingleDSC objects
        self.single_mesurement = [SingleDSC(
            file_dir=self.file_dir, file_name=file) for file in files]

    def data_cleaner(self, **kwargs):
        """Clean the data"""
        for single in self.single_mesurement:
            single.data_cleaner(**kwargs)

        # now concatenate all the data in a single dataframe called self.data
        # here we use Temperture as index amd we concatenate the Heat Flow
        self.data = concat([single.data.set_index("Temperature (C)")[["Heat Flow (W/g)"]]
                            for single in self.single_mesurement], axis=1)

        self.data = self.data.dropna()

    def data_saver(self, output_dir="output_data", name="output_data.txt"):
        """Save the data"""
        if not path.exists(output_dir):
            makedirs(output_dir)

    def data_plotter(self, **kwargs):
        """Plot the data"""
        for i, single in enumerate(self.single_mesurement):
            single.data_plotter(
                output_dir=kwargs["output_dir"], name=kwargs["name"] + str(i))

    def data_check(self, **kwargs):
        for single in self.single_mesurement:
            single.data_check(**kwargs)





def main():
    # the main works in this way, you call python -m lab single_dsc --file_dir path
    # it cleans tha data and save it in the output directory

    import argparse

    parser = argparse.ArgumentParser(
        description="Clean a DSC file and plots it")

    parser.add_argument("--file_dir", type=str,
                        help="The directory of the file")
    parser.add_argument("--file_name", type=str, help="The name of the file")
    parser.add_argument("--output_dir", type=str,
                        help="The directory of the output file")
    parser.add_argument("--output_name", type=str,
                        help="The name of the output file")
    parser.add_argument("--min_temp", type=float,
                        help="The minimum temperature")
    parser.add_argument("--max_temp", type=float,
                        help="The maximum temperature")
    parser.add_argument("--p", type=float,
                        help="The p parameter for the baseline")
    parser.add_argument("--lam", type=float,
                        help="The lambda parameter for the baseline")
    parser.add_argument("--niter", type=int,
                        help="The number of iterations for the baseline")

    args = parser.parse_args()

    single = SingleDSC(file_dir=args.file_dir, file_name=args.file_name)

    single.data_reader()

    single.data_cleaner(min_temp=args.min_temp, max_temp=args.max_temp,
                        p=args.p, lam=args.lam, niter=args.niter)

    single.data_saver(output_dir=args.output_dir, name=args.output_name)

    single.data_plotter(output_dir=args.output_dir, name=args.output_name)


if __name__ == "__main__":
    main()
