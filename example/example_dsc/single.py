from lab import *
import os


def main():
    """Main implementation to quickly test the code"""

    # READ THE DATA

    # tell where the raw data is
    dsc = SingleDSC(
        "input_data/2perc/231116_RBX_RSO_2perc_A_1st_heating_raw_data.txt"
    )
    print(dsc.data)

    # CLEAN THE DATA AND SAVE IT
    dsc.data_cleaner()

    # save the cleaned data
    output_dir = "output_data/single"
    dsc.data_saver(output_dir=output_dir, name="cleaned_data.txt")
    print(dsc.data)

    # PLOT THE DATA
    dsc.data_plotter(output_dir=output_dir, name="plot_single.png")


if __name__ == "__main__":
    main()
