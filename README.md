# Speech To Text

This project performs audio transcriptions using the Whisper model and processes the results to generate text transcriptions.

## Folder Structure

- `src/`: Contains files related to the main script and auxiliary functions.
- `src/commons/`: Contains shared functions and tools.
- `speech_to_text.py`: The main script that performs audio transcriptions and processing.
- `transcript.py`: Orchestrates the execution of the main script from the `src` folder.

## Usage Instructions

1. Clone this repository to your local machine:
git clone https://github.com/gamug/speec2text.git

2. Install the required dependencies:
pip install -r requirements.txt

3. Run the main script:
python transcript.py

## Dependencies

- WhisperX
- pandas
- torch
- pydub

Specific dependency versions can be found in the `requirements.txt` file.