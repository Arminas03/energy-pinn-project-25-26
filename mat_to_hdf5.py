import pandas as pd
from scipy.io import loadmat
import h5py


def get_charge_char_data(charge_char_mat_data):
    return {
        var: charge_char_mat_data[var].T[0] for var in charge_char_mat_data.dtype.names
    }


def get_cycle_data(cycle_mat_data):
    return {
        charge_char: get_charge_char_data(cycle_mat_data[charge_char][0][0])
        for charge_char in cycle_mat_data.dtype.names
    }


def get_data_for_cell(cell_mat_data):
    return {
        cycle: get_cycle_data(cell_mat_data[cycle][0][0])
        for cycle in cell_mat_data.dtype.names
    }


def get_formatted_data(mat_path):
    mat_data = loadmat(mat_path)
    num_cells = 8

    return {
        cell: get_data_for_cell(mat_data[cell][0][0])
        for cell in [f"Cell{i+1}" for i in range(num_cells)]
    }


def flatten_data_to_variables(data):
    return {
        (cell, cycle, charge_char): pd.DataFrame(var_data)
        for cell, cycle_data in data.items()
        for cycle, char_data in cycle_data.items()
        for charge_char, var_data in char_data.items()
    }


def save_data_as_h5(data):
    f = h5py.File("data_Oxford.hdf5", "w")

    flat_data = flatten_data_to_variables(data)

    for cell, cycle, char in flat_data:
        f.create_dataset(f"{cell}/{cycle}/{char}", data=flat_data[(cell, cycle, char)])


def main(mat_path):
    data = get_formatted_data(mat_path)

    save_data_as_h5(data)


if __name__ == "__main__":
    path_to_mat_file = "Oxford_Battery_Degradation_Dataset_1.mat"

    main(path_to_mat_file)
