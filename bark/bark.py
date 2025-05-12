from transformers import AutoProcessor, BarkModel
import scipy

processor = AutoProcessor.from_pretrained("suno/bark")
model = BarkModel.from_pretrained("suno/bark")

def generate_bark(tts_message, voice_preset, output_filename):
    # get the device
    inputs = processor(tts_message, voice_preset=voice_preset)

    # generate the audio
    audio_array = model.generate(**inputs)
    audio_array = audio_array.cpu().numpy().squeeze()

    # output the audio
    sample_rate = model.generation_config.sample_rate
    scipy.io.wavfile.write(output_filename, rate=sample_rate, data=audio_array)
