import os
import utils
import multiprocessing
import math
import sys


input_path = "songs"
wavs = os.listdir('raw/'+input_path)
# wavs = ["说书人_vocals.wav", "说书人_vocals_enhanced.wav"]
# speaker_dict = {"billy300": "billy300"}
# speaker_dict = {model: speaker}
wavs = [wav for wav in wavs if wav.endswith(".wav")]
# wavs = wavs[1000:2000]
wavs = ["深夜的歌_马陈_vocals_enhanced.wav"]

# speaker_dict = {"mozuoyi_perturb": "mozuoyi", "mozuoyi_noise": "mozuoyi"}
speaker_dict = {"mofei_150": "mofei"}

print(wavs)
def process_one(model, speaker, wav):
    # cmd = "CUDA_VISIBLE_DEVICES=7 python inference_main.py -m {3} \
    #     -c logs/{0}/config.json -n {2} -t 0 -s {1} -wf wav -i {4} -shd -f0p dio -dm logs/{0}/diffusion/model_100000.pt -dc logs/{0}/diffusion/config.yaml".format(model, speaker, wav, utils.latest_checkpoint_path(os.path.join("logs", model), "G_*.pth"), input_path)
    cmd = "CUDA_VISIBLE_DEVICES=1 python inference_main.py -m {3} \
        -c logs/{0}/config.json -n {2} -t 0 -s {1} -wf wav -i {4} -f0p dio".format(model, speaker, wav, utils.latest_checkpoint_path(os.path.join("logs", model), "G_*.pth"), input_path)
    os.system(cmd)
    print(cmd)

def process_batch(chunk):
    print(chunk)
    for model, speaker, wav in chunk:
        process_one(model, speaker, wav)
        

if __name__ == '__main__':
    multiprocessing.set_start_method('spawn',force=True)
    processes = []
    chunks = []

    
    for wav in wavs:
        for model, speaker in speaker_dict.items():
            # print(model, speaker)
            chunks.append((model, speaker, wav))

    num_processes = 6
    chunk_size = int(math.ceil(len(chunks) / num_processes))
    chunks = [chunks[i:i + chunk_size] for i in range(0, len(chunks), chunk_size)]
    print([len(c) for c in chunks], chunks)
    processes = [multiprocessing.Process(target=process_batch, args=(chunk,)) for chunk in chunks]
    for p in processes:
        p.start()
        
