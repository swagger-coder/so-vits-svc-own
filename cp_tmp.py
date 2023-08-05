import os,sys

raw_dir = '/mnt/nfs/dev-aigc-27/data1/mqjiang/data/mvts'
spk_lst = ['liuneng/wav_ori', 'henanmoxiaomiao/交付', 'mowanzhu/wav_ori', 'dora_sample/48k_sample', 'moxiaoxi/wav_ori']
out_dir = './raw/fangyan'
os.makedirs(out_dir, exist_ok=True)
# copy 10句
for spk in spk_lst:
    spk_dir = os.path.join(raw_dir, spk)
    wav_lst = os.listdir(spk_dir)
    for wav in wav_lst[:10]:
        wav_path = os.path.join(spk_dir, wav)
        os.system(f'cp {wav_path} {out_dir}')