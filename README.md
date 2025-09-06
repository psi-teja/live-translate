# live-translate

A real-time speech translation tool using OpenAI Whisper for speech recognition and Facebook NLLB for translation.

## Features

- Real-time audio capture and transcription
- Seamless translation between 200+ languages (via NLLB)
- GPU acceleration support (Apple Silicon MPS)
- Simple command-line interface

## Requirements

- Python 3.8+
- [PyTorch](https://pytorch.org/)
- [transformers](https://huggingface.co/docs/transformers/index)
- [openai-whisper](https://github.com/openai/whisper)
- [pyaudio](https://people.csail.mit.edu/hubert/pyaudio/)
- numpy

Install dependencies:

```sh
pip install torch transformers openai-whisper pyaudio numpy
```

## Usage

Run the translator with default settings (English to Spanish):

```sh
python main.py
```

To change source or target languages, edit the following lines in `main.py`:

```python
translator = RealTimeTranslator(source_lang="eng_Latn", target_lang="spa_Latn")
```

Refer to the [NLLB language codes](https://huggingface.co/facebook/nllb-200-distilled-600M#languages) for supported languages.

## How it works

1. Captures audio from your microphone in real-time.
2. Transcribes speech to text using Whisper.
3. Translates the transcribed text using Facebook NLLB.
4. Prints both the original and translated text to the console.

## Notes

- For best performance, use a machine with a compatible GPU (Apple Silicon MPS is supported).
- Press `Ctrl+C` to stop the translation.

##