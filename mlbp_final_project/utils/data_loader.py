import numpy as np
import os

__PATH_TO_THIS_FILE__ = os.path.dirname(os.path.realpath(__file__))
__DATA_FOLDER__ = os.path.join(__PATH_TO_THIS_FILE__, os.path.pardir, "data")
__OUTPUT_FOLDER__ = os.path.join(__PATH_TO_THIS_FILE__, os.path.pardir, os.path.pardir, "output")

def get_output_folder():
    return __OUTPUT_FOLDER__

def get_data_folder():
    return __DATA_FOLDER__

def file_loader(filename, skiprows=0):
    data_path = os.path.join(__DATA_FOLDER__, filename)
    data = np.loadtxt(data_path, dtype=np.double, delimiter=',', skiprows=skiprows)
    return data

def load_test_data():
    return file_loader("test_data.csv")

def load_train_data():
    return file_loader("train_data.csv")

def load_train_labels():
    return file_loader("train_labels.csv")

def load_dummy_solution_accuracy():
    return file_loader("dummy_solution_accuracy.csv", skiprows=1)

def load_dummy_solution_logloss():
    return file_loader("dummy_solution_logloss.csv", skiprows=1)

def save_csv_output(data, file_name, allow_overwrite=False):
    """Saves the output.

    If the file already exists appends a unique identifier to the name.

    Parameters
    ----------
    data: numpy.array
    file_name
    allow_overwrite

    Returns
    -------

    """
    ext = "csv"
    file_path = os.path.join(__OUTPUT_FOLDER__, "{}.{}".format(file_name, ext))

    i = 0
    while os.path.exists(file_path) and not allow_overwrite:
        print("{} already exists, retrying..".format(file_path))
        i += 1
        renamed_file_name = "{}_{}_.{}".format(file_name, i, ext)
        file_path = os.path.join(__OUTPUT_FOLDER__, "{}.{}".format(renamed_file_name, ext))

    with open(file_path, "w") as csv_file:

        # Write the header.
        if len(data.shape) > 1:
            csv_file.write("Sample_id,Class_1,Class_2,Class_3,Class_4,Class_5,Class_6,Class_7,Class_8,Class_9,Class_10\n")  # noqa
        else:
            csv_file.write("Sample_id,Sample_label\n")

        # Write the data.
        for i in range(data.shape[0]):
            line = i + 1
            data_line = data[i]

            # Check if it's a multi column.
            if len(data.shape) > 1:
                data_line_str_list = [str(x) for x in data_line]
                data_line_str = ",".join(data_line_str_list)
            else:
                data_line_str = str(data_line)

            csv_file.write("{},{}\n".format(line, data_line_str))

    print("Saved: {}".format(file_path))
