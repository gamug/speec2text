import os
from src.speech_to_text import (
    parameters, separateChannels, whisperTranscription,
    concatTranscript, exportText
    )
from src.commons.common_tools import check_directories

def main():
    check_directories(parameters)

    audioFile = os.path.join(parameters['audios'], "test_audio_1.mp3")
    speakersPath = os.path.join(parameters['speakers'], "test_audio_1")

    cliente, asesor = separateChannels(audioFile, speakersPath)

    resultTranscriptCliente = whisperTranscription(cliente)
    resultTranscriptAsesor = whisperTranscription(asesor)

    df_transcript = concatTranscript(resultTranscriptCliente, resultTranscriptAsesor)
    exportText(df_transcript, parameters['transcription'], "transcript_1.txt")

if __name__ == "__main__":
    main()