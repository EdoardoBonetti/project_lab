from lab import *


def main():
    """Main implementation to quickly test the code"""
    dsc = SingleDSC("data/DSC_1.txt")
    dsc.data_reader()
    dsc.data_cleaner()
    dsc.data_plotter()


if __name__ == "__main__":
    main()
