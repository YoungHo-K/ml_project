import os
import glob
import platform


def retrieve(input_dir_path):
    input_dir_path = os.path.join(input_dir_path, "*")
    input_dir_list = glob.glob(input_dir_path)

    file_list = list()
    for path in input_dir_list:
        if os.path.isdir(path):
            file_list += retrieve(path)

        else:
            file_list.append(path)

    return file_list


def get_file_name(file_path):
    file_name = os.path.basename(file_path)
    file_name = os.path.splitext(file_name)[0]

    return file_name


def get_extension(file_path):
    return os.path.splitext(file_path)[1]


def read_file_to_bytes(file_path):
    if not os.path.isfile(file_path):
        raise Exception("[ERROR] Invalid file path.")

    with open(file_path, "rb") as file_descriptor:
        bytes_ = file_descriptor.read()

    if platform.system() == "Windows":
        byte_sequence = list()
        for byte in bytes_:
            byte_sequence.append(int(f"{byte:02X}", 16))

        return byte_sequence

    if platform.system() == "Linux":
        return list(bytes_)

    raise Exception(f"[ERROR] Undefined system: {platform.system()}")

