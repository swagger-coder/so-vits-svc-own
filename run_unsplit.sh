data_name=$1 # e.g. luo
ngpu=$2 # e.g. 0/0,1,2
port=$3 # e.g. 29500

# Step 1: audio slice
# save in dataset_raw/${data_name}
python audio_slice.py ${data_name}

# Step 2: resample
# save in dataset/44k/${data_name}
if [ -d "dataset/44k/${data_name}" ]; then
    rm -rf dataset/44k/${data_name}
fi
python resample.py --in_dir ./dataset_raw/${data_name}

if [ -d "./dataset_raw_done/${data_name}" ]; then
    rm -rf  ./dataset_raw_done/${data_name}
fi
mv ./dataset_raw/${data_name} ./dataset_raw_done/
if [ ! -d "dataset/${data_name}" ]; then
    mkdir dataset/${data_name}
    cp -r  dataset/44k/${data_name} dataset/${data_name}/${data_name}
fi


# # Step 3: preprocess
python preprocess_flist_config.py  --speech_encoder vec768l12 --data_name ${data_name} --source_dir dataset/${data_name}
CUDA_VISIBLE_DEVICES=${ngpu} python preprocess_hubert_f0.py --in_dir dataset/${data_name} --f0_predictor dio

if [ ! -d "logs/${data_name}" ]; then
    mkdir logs/${data_name}
    cp logs/base/* logs/${data_name}/
fi

cp configs/config_${data_name}.json logs/${data_name}/config.json
CUDA_VISIBLE_DEVICES=${ngpu} python train.py -c configs/config_${data_name}.json -m ${data_name} -p ${port}