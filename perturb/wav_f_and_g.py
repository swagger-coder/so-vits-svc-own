import functional
import librosa
import torch
import tqdm
import numpy as np
import scipy
from concurrent.futures import ThreadPoolExecutor
import scipy.io.wavfile
import os
import soundfile as sf
import sys

def save_wav(wav, path, sr):
    # scipy.io.wavfile.write(path, sr, wav.astype(np.int16))
    # librosa.output.write_wav(path, wav, sr)
    sf.write(path, wav, sr)

def perturb(wav, sr, type):
    if type == "f":
        return functional.f(wav, sr)
    elif type == "g":
        return functional.f(wav, sr)
    elif type == "formant":
        sound = functional.wav_to_Sound(wav.numpy(), sampling_frequency=sr)
        return functional.formant_shift(sound).values.squeeze(0)
    elif type == "formant_and_pitch":
        sound = functional.wav_to_Sound(wav.numpy(), sampling_frequency=sr)
        return functional.formant_and_pitch_shift(sound).values.squeeze(0)
    elif type == "peq":
        return functional.parametric_equalizer(wav, sr).numpy()
    else:
        raise ValueError("bad type")


def load_wav(path: str, sr: int = None):
    assert sr is not None
    try:
        wav_numpy, sr = librosa.core.load(path, sr=sr)
        # print ("wav_numpy ,", wav_numpy[5000])
        wav_torch = torch.from_numpy(wav_numpy).float()
    except:
        raise ValueError(f"could not load audio file from path :{path}")
    return wav_numpy, wav_torch

def process_sentence(fname, input_path, sr, output_path):
    perturb_types = ["f", "g", "formant", "formant_and_pitch", "peq"]
    # _, wav_torch = load_wav(os.path.join(input_path, fname), sr)
    _, wav_torch = load_wav(input_path, sr)
    for perturb_type in perturb_types:
        wav_perturb = perturb(wav_torch, sr, perturb_type)
        new_fname = os.path.splitext(fname)[0] + "_after_" + perturb_type + ".wav"
        save_wav(wav_perturb, os.path.join(output_path, new_fname), sr)



def process_sentences(fnames, input_path, sr, output_path, nprocs: int = 1):
    if nprocs == 1:
        for p in fnames:
            process_sentence(p, input_path[p], sr, output_path)
    else:
        with ThreadPoolExecutor(nprocs) as pool:
            futures = []
            with tqdm.tqdm(total=len(fnames)) as progress:
                for p in fnames:
                    future = pool.submit(process_sentence, p, input_path[p], sr, output_path)
                    future.add_done_callback(lambda p: progress.update())
                    futures.append(future)


if __name__ == "__main__":
    sr = 16000
    # path = "/export/expts2/mingqi.jiang/workspace/vc_github/NANSY/source_wav"
    #wav_scp = "/export/expts2/mingqi.jiang/wangxu_work/vc/deep-voice-conversion-ar8/post_am/multi_spk_34/wav.scp"
    wav_scp = sys.argv[1]
    fnames=[]
    paths=dict()
    with open(wav_scp, "r") as f:
        for line in f.readlines():
            pieces=line.strip("\n").split()
            fnames.append(pieces[0])
            paths[pieces[0]] = pieces[1]
    print (fnames[5])
    print (paths[fnames[5]])


    #output_path = "/export/expts2/mingqi.jiang/workspace/vc_github/NANSY/all_perturb"
    output_path = sys.argv[2]
    # fnames = os.listdir(path)
    # paths = [os.path.join(path, fnames)]

    process_sentences(fnames, paths, sr, output_path, nprocs=10)
    
    # perturb_types = ["f", "g", "formant", "formant_and_pitch", "peq"]
    # for fname in fnames:
    #     wav_numpy, wav_torch = load_wav(os.path.join(path, fname), sr)
    #     for perturb_type in perturb_types:
    #         print (f'{fname} {perturb_type}')
    #         print ("test wav, ", wav_torch[5000])
    #         wav_perturb = perturb(wav_torch, sr, perturb_type)
    #         print ("test, ", wav_perturb[5000])
    #         new_fname = os.path.splitext(fname)[0] + "after_" + perturb_type + ".wav"
    #         save_wav(wav_perturb, os.path.join(output_path, new_fname), sr)
