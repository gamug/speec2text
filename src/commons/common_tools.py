import os, sys
import pandas as pd, numpy as np


parameters = {
    'path': os.path.dirname(sys.path[0]),
    'path_in': os.path.join(os.path.dirname(sys.path[0]), '01_data'),
    'parametric': os.path.join(os.path.dirname(sys.path[0]), '01_data', 'parametrics'),
    'path_out': os.path.join(os.path.dirname(sys.path[0]), '03_output'),
    'curated': os.path.join(os.path.dirname(sys.path[0]), '01_data', 'curated'),
    'matching': os.path.join(os.path.dirname(sys.path[0]), '01_data', 'matching_dbs'),
    'topic_modeling': os.path.join(os.path.dirname(sys.path[0]), '03_output', 'topic_modeling'),
    'speech': os.path.join(os.path.dirname(sys.path[0]), '03_output', 'speech_to_text'),
    'audios': os.path.join(os.path.dirname(sys.path[0]), '01_data', 'audios_ccenter'),
    'speakers': os.path.join(os.path.dirname(sys.path[0]), '03_output', 'speech_to_text', 'audios_speakers'),
    'transcription': os.path.join(os.path.dirname(sys.path[0]), '03_output', 'speech_to_text', 'transcription')
}

def check_directories(dict_):
    """
    Check and create directories if they do not exist.

    Parameters
    ----------
    dict_ : dict
        A dictionary containing directory paths to be checked and created if necessary.

    Notes
    -----
    This function checks the existence of directories specified in the provided dictionary. If a directory does not exist,
    it creates the directory using the `os.mkdir()` function. The dictionary keys represent the names of the directories,
    and the values represent the corresponding paths.
    """
    for path in dict_.values():
        if not os.path.exists(path):
            os.mkdir(path)