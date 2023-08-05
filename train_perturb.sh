data_name=$1
ngpu=$2
port=$3
# 重采样，将原始数据集重采样为44k
# python resample.py --in_dir ./dataset_raw/${data_name} --skip_loudnorm
# mv ./dataset_raw/${data_name} ./dataset_raw_done/
# if [ ! -d "dataset/${data_name}" ]; then
#     mkdir dataset/${data_name}
# fi
# mv dataset/44k/${data_name} dataset/${data_name}/${data_name}


# python preprocess_flist_config.py  --speech_encoder vec768l12 --data_name ${data_name} --source_dir dataset/${data_name}

# python preprocess_hubert_f0.py --in_dir dataset/${data_name} --f0_predictor dio --use_diff


# if [ ! -d "logs/${data_name}" ]; then
#     mkdir logs/${data_name}
# fi
# cp logs/base_new/* logs/${data_name}/
# cp configs/config_${data_name}.json logs/${data_name}/config.json
CUDA_VISIBLE_DEVICES=${ngpu} python train_perturb.py -c configs/config_${data_name}.json -m ${data_name} -p ${port}