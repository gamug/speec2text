import whisperx
import gc 
import pandas as pd
import torch
import warnings
import os, sys

from pydub import AudioSegment
from src.commons.common_tools import check_directories, parameters

warnings.filterwarnings("ignore")

# Define device and compute type for Whisper model
device = "cuda" 
compute_type = "float16" # change to "int8" if low on GPU mem (may reduce accuracy)
batch_size = 16 # reduce if low on GPU mem

def separateChannels(file, savePath):
    """
    Separate a stereo audio file into two mono channels and export as MP3 files.

    Parameters
    ----------
    file : str
        Path to the stereo audio file.
    savePath : str
        Path to the directory where separated mono audio files will be saved.

    Returns
    -------
    str, str
        Paths to the separated mono audio files.
    """
    stereoAudio = AudioSegment.from_file(file, format="mp3")

    monoAudios = stereoAudio.split_to_mono()
    
    audioRightFile = f"cliente_{file}"
    audioLeftFile = f"asesor_{file}"

    os.makedirs(savePath, exist_ok=False)

    monoAudios[0].export(os.path.join(savePath, audioLeftFile), format="mp3")
    monoAudios[1].export(os.path.join(savePath, audioRightFile), format="mp3")
    
    return os.path.join(savePath, audioLeftFile), os.path.join(savePath, audioRightFile)
    
def whisperTranscription(file):
    """
    Perform speech transcription using the Whisper model.
    
    Parameters
    ----------
    file : str
        Path to the audio file for transcription.
    
    Returns
    -------
    dict
        Transcription results including segments, start times, end times, and transcribed text.
    """
    model = whisperx.load_model("large-v2", device, compute_type=compute_type)

    audio = whisperx.load_audio(file)
    result = model.transcribe(audio, batch_size=batch_size, language="es")

    gc.collect(); torch.cuda.empty_cache(); del model
    
    model_a, metadata = whisperx.load_align_model(language_code=result["language"], device=device)
    result = whisperx.align(result["segments"], model_a, metadata, audio, device, return_char_alignments=False)
    return result

def filterModelMistakes(df):
    """
    Filter transcribed segments based on text length and duration.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing transcribed segments with timing and text information.

    Returns
    -------
    pandas.DataFrame
        DataFrame with additional columns for timing analysis and validity checks.
    """
    df = df.assign(T_DELTA=df.end-df.start)
    df = df.assign(WORD_LENGHT=df.text.str.split(' {1,}').str.len())
    df = df.assign(RATIO=df.WORD_LENGHT/df.T_DELTA)
    df = df.assign(VALID=df.RATIO<20)
    return df

def concatTranscript(asesor, cliente):
    """
    Concatenates and processes transcriptions of advisor and client.

    Parameters
    ----------
    asesor : dict
        Transcription of the advisor.
    cliente : dict
        Transcription of the client.

    Returns
    -------
    pandas.DataFrame
        Concatenated and processed transcriptions.
    """
    transcriptCliente = pd.DataFrame.from_dict(cliente['segments'])
    transcriptCliente = transcriptCliente.assign(speaker='cliente')
    transcriptCliente[['start', 'end', 'text', 'speaker']]

    transcriptAsesor = pd.DataFrame.from_dict(asesor['segments'])
    transcriptAsesor = transcriptAsesor.assign(speaker='asesor')
    transcriptAsesor[['start', 'end', 'text', 'speaker']]

    concat = pd.concat([transcriptCliente, transcriptAsesor])[['start', 'end', 'text', 'speaker']].sort_values('start')

    return filterModelMistakes(concat)

def exportText(df, filePath, fileName):
    """
    Export transcribed text to a text file.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing transcribed segments with speaker and text information.
    filePath : str
        Path to the directory where the text file will be saved.
    fileName : str
        Name of the text file.

    Returns
    -------
    None
    """
    with open(os.path.join(filePath, fileName), 'w') as file:
        for i, row in df.iterrows():
            line = f"{row['speaker']}: {row['text']}\n"
            file.write(line)    