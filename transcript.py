import os
import json
from src.speech_to_text import (
    separateChannels, whisperTranscription,
    concatTranscript, exportText
    )
from src.commons.common_tools import check_directories, parameters

def main():
    check_directories(parameters)

    with open(os.path.join(parameters['parametric'], 'speech2text.json'), 'r') as f:
        parametric = json.loads(f.read())

    nameFolder = '.'.join(parametric['audio_file'].split('.')[:-1])

    audioFile = os.path.join(parameters['audios'], parametric['audio_file'])
    speakersPath = os.path.join(parameters['speakers'], nameFolder)
    cliente, asesor = separateChannels(audioFile, speakersPath)

    resultTranscriptCliente = whisperTranscription(cliente)
    resultTranscriptAsesor = whisperTranscription(asesor)

    df_transcript = concatTranscript(resultTranscriptCliente, resultTranscriptAsesor)
    exportText(df_transcript, parameters['transcription'], parametric['transcript_file'])

if __name__ == "__main__":
    main()