import pickle
import os
import json


def write_to_disk(data, file_name, folder_name="",
                  is_json=False, is_file=False, is_object=False):
    """
    :param data: any data type to save to disk
    :param file_name: any extension (.dat or .p in general, or anything else)
    :param folder_name: can leave empty (str, optional)
    :param is_json: to indicate whether the file is a json file or not
    :param is_file: to indicate whether the file is a human readable file or not
    :param is_object: to indicate whether the data should be pickled as a binary python object
    :return: None (but saves to disk)
    """
    arguments_sum = sum([is_json, is_file, is_object])
    if arguments_sum == 0:
        raise ValueError("Have to choose exactly one file mode!")
    elif arguments_sum > 1:
        raise ValueError("Cannot activate several file modes at the same time. Have to choose only one!")

    file = os.path.join(folder_name, file_name)
    if os.path.isfile(file):
        os.remove(file)

    if is_json:
        with open(file, "w") as f:
            json.dump(data, f)
    elif is_file:
        with open(file, "w") as f:
            f.write(data)
    elif is_object:
        with open(file, "wb") as f:
            pickle.dump(data, f)
    else:
        raise Exception("Unexpected error. Doesn't make sense, check source code...")


def load_from_disk(path, is_json=False, is_file=False, is_object=False):
    """
    :param path: os.path.join() path
    :param is_json: whether the file is a json file or not
    :param is_file: to indicate whether the file is a human readable file or not
    :param is_object: to indicate whether the file is pickled as a binary python object
    :return: loaded data object
    """
    try:
        if is_json:
            with open(path) as f:
                data = json.load(f)
        elif is_file:
            with open(path) as f:
                data = f.read()
        elif is_object:
            with open(path, "rb") as f:
                data = pickle.load(f)
        else:
            raise Exception("Unexpected error. Doesn't make sense, check source code...")
    except FileNotFoundError:
        data = None
    return data


if __name__ == "__main__":
    pass
