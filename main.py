import sys
import os
import argparse
sys.path.append(os.path.join(os.path.dirname(__file__), 'seed_vc'))
from xtts.xtts import generate_xtts
from bark.bark import generate_bark
from seed_vc.inference import main as vc_main

# the texts to be converted to audio
texts = [
    "Jump.",
    "Its snowing.",
    "She is singing beautifully.",
    "They built a tree house last summer.",
    "My brother studies engineering at a university abroad.",
    "We went hiking early in the morning and returned before sunset.",
    "He apologized quickly after realizing his mistake during the meeting.",
    "Although it was expensive, she bought the camera because it had excellent reviews.",
    "Before making a final decision, he carefully considered everyone's opinions and read through all the available research.",
    "She gathered all the necessary documents, submitted her application online, and waited patiently for the results to be officially announced."
]

# the speaker file to be used for xtts and seedvc
speaker_filename = "peter.wav"

# the language to be used for tts
language = "en"

for i, text in enumerate(texts):
    # filename for the generated audio
    xtts_output_filename = f"xtts_out_{i}.wav"

    # generate audio using xtts
    generate_xtts(text, speaker_filename, language, xtts_output_filename)

    print(f"Xtts audio saved to {xtts_output_filename}")

for i, text in enumerate(texts):
    # filename for the generated audio
    bark_output_filename = f"bark_out_{i}.wav"

    voice_preset = "v2/en_speaker_6"

    # generate audio using xtts
    generate_bark(text, voice_preset, bark_output_filename)

    print(f"Bark audio saved to {bark_output_filename}")

for i, text in enumerate(texts):

    # filename for the generated audio
    seedvc_output_filename = f"seedvc_out_{i}.wav"

    args = argparse.Namespace(
        source=f"./bark_outputs/bark_out_{i+1}.wav",
        target=speaker_filename,
        output=seedvc_output_filename,
        diffusion_steps=25,
        length_adjust=1.0,
        inference_cfg_rate=0.7,
        f0_condition=False,
        auto_f0_adjust=False,
        semi_tone_shift=0,
        checkpoint=None,
        config=None,
        fp16=True
    )

    filename = f"bark_with_seedvc_outputs"

    vc_main(args)
