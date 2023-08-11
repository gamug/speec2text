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

3. Requires the command-line tool ffmpeg to be installed on your system, which is available from most package managers:
 on Ubuntu or Debian
 sudo apt update && sudo apt install ffmpeg

 on Arch Linux
 sudo pacman -S ffmpeg

 on MacOS using Homebrew (https://brew.sh/)
 brew install ffmpeg

 on Windows using Chocolatey (https://chocolatey.org/)
 choco install ffmpeg

 on Windows using Scoop (https://scoop.sh/)
 scoop install ffmpeg

4. Run the main script:
python transcript.py

## Dependencies

- Whisper
- pandas
- torch
- pydub

Specific dependency versions can be found in the `requirements.txt` file.
More information about model: https://github.com/openai/whisper