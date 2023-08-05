import librosa  # Optional. Use any library you like to read audio files.
import soundfile  # Optional. Use any library you like to write audio files.
import os
import sys

from slicer2 import Slicer



def process(wav_path, name, spk):
    out_path = f"./dataset_raw/{spk}"
    if not os.path.exists(out_path):
        os.makedirs(out_path)
    name = name.replace(".", "").replace("(", "_").replace(")", "").replace("'", "").replace(" ", "")
    # name = wav_path.split("_")[1][0]
    print(name)
    audio, sr = librosa.load(wav_path, sr=None, mono=False)  # Load an audio file with librosa.
    slicer = Slicer(
        sr=sr,
        threshold=-40,
        min_length=5000,
        min_interval=300,
        hop_size=10,
        max_sil_kept=500
    )
    print("begin")
    chunks = slicer.slice(audio)
    for i, chunk in enumerate(chunks):
        print(i)
        if len(chunk.shape) > 1:
            chunk = chunk.T  # Swap axes if the audio is stereo.
        soundfile.write(f'{out_path}/{name}_{i}.wav', chunk, sr)  # Save sliced audio files with soundfile.

if __name__ == '__main__':
    path = "./dataset_raw_unsplit"
    spk = sys.argv[1]
    path = os.path.join(path, spk)
    wavs = os.listdir(path)
    wavs = [i for i in wavs if i.endswith("WAV") or i.endswith("wav")]
    print(wavs)
    for i,wav in enumerate(wavs):
        wav_path = os.path.join(path, wav)
        process(wav_path, wav[:-4], spk)
