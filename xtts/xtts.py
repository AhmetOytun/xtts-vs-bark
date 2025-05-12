import torch
from torch.serialization import add_safe_globals
from TTS.api import TTS
from TTS.tts.configs.xtts_config import XttsConfig,XttsAudioConfig,XttsArgs
from TTS.tts.configs.shared_configs import BaseDatasetConfig

def generate_xtts(tts_message, speaker_filename, tts_language, output_filename):
    add_safe_globals({XttsConfig, XttsAudioConfig, BaseDatasetConfig, XttsArgs})
    
    # get the device
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # load the model
    tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)

    # generate the audio
    wav = tts.tts(text=tts_message,speaker_wav=speaker_filename, language=tts_language)

    # output the audio
    tts.tts_to_file(text=tts_message,speaker_wav=speaker_filename, language=tts_language, file_path=output_filename)
    return wav