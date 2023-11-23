from pathlib import Path
import pickle
from typing import Union


def save_data(path_or_filename: Union[str, Path], data: object):
    """Saves data to a file via `pickle`.

    Args:
        path_or_filename (Union[str, Path]): Path or filename of the outputfile e.g. output.dat.
        data (object): Object to be saved.
    """

    path = Path(path_or_filename)
    # Make sure parents directories exist
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "wb") as file:
        pickle.dump(data, file)
        print("Data saved at " + str(path))


def load_data(path_or_filename: Union[str, Path]):
    """Loads data object via `pickle`.

    Args:
        path_or_filename (Union[str, Path]): Path or filename of the data file e.g. output.dat.

    Returns:
        object: The data object.
    """

    with open(path_or_filename, "rb") as file:
        data = pickle.load(file)

        return data
