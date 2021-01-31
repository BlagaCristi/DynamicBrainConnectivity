import array
import os
import struct

from util.constants import PARSED_DATA, SUBJECT_FILE_EXTENSION
from util.util import get_string_from_number


def read_value_from_binary_file(file, unpack_type, type_size):
    """
    Reads a value from a binary file.
    """
    read_value = file.read(type_size)
    return struct.unpack(unpack_type, read_value)[0]


def read_array_from_binary_file(file, unpack_type, type_size, array_size):
    """
    Read a chunk from a binary file ALL AT ONCE.
    """

    numbers = array.array(unpack_type)
    numbers.fromfile(file, array_size)
    numbers = numbers.tolist()

    return numbers


# read values from a binary file AT ONCE in one operation AS A CHUNK
# unpack_type = the type of the value read
# type_size = the size of the read type
# array_size = the total number of values to be read
def read_array_from_unopened_binary_file(file, unpack_type, type_size, array_size):
    with open(file, "rb") as f:
        numbers = f.read(type_size * array_size)
        float_array_string = struct.unpack(unpack_type * array_size, numbers)

    return float_array_string


# read all the values from a binary file ONE BY ONE
def read_values_from_binary_file_one_by_one(file, unpack_type, type_size):
    numbers = []

    with open(file, "rb") as f:
        read_value = f.read(type_size)
        while read_value:
            number = struct.unpack(unpack_type, read_value)[0]
            numbers.append(number)

            read_value = f.read(type_size)

    return numbers


# write an array of values to a binary file AT ONCE in one operation AS A CHUNK
# subject_number = the number of the subject that will be used when creating the file name
# values = an array of float values
# length = the size of the array (how many values to be written)
def write_to_binary_file(subject_number, values, length, output_directory):
    # go to the directory where to save the data
    parsed_data_directory = os.path.join(output_directory, PARSED_DATA)
    if not os.path.exists(parsed_data_directory):
        os.makedirs(parsed_data_directory)

    # open file as 'subject-number.bin'
    trial_file = get_string_from_number(subject_number) + SUBJECT_FILE_EXTENSION
    trial_file = os.path.join(parsed_data_directory, trial_file)

    with open(trial_file, "wb+") as file:
        float_array_string = struct.pack('f' * length, *values)
        file.write(float_array_string)


def read_chunk_from_binary_file(file_path, start, chunk_size, type_size, unpack_type):
    file = open(file_path, "rb")

    # move the file pointer to the beginning of the chunk
    file.seek(start * type_size)

    # read the chunk
    numbers = file.read(type_size * chunk_size)

    # convert from binary to float
    float_array = struct.unpack(unpack_type * chunk_size, numbers)

    file.close()
    return list(float_array)