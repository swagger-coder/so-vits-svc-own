data_name=$1
ngpu=$2
port=$3
# 重采样，将原始数据集重采样为44k
# python resample_multi.py --in_dir /mnt/nfs/dev-aigc-28/data3/pengfeiyue/base_new --out_dir2 /mnt/cloud/nfs/pengfeiyue/svc_dataset/base_210  --skip_loudnorm
# # mv ./dataset_raw/${data_name} ./dataset_raw_done/
# if [ ! -d "dataset/${data_name}" ]; then
#     mkdir dataset/${data_name}
# fi
# mv dataset/44k/${data_name} dataset/${data_name}/${data_name}


# python preprocess_flist_config.py  --speech_encoder vec768l12 --data_name base_210 --source_dir /mnt/cloud/nfs/pengfeiyue/svc_dataset/base_210

#CUDA_VISIBLE_DEVICES=0 python preprocess_hubert_f0.py --in_dir /mnt/cloud/nfs/pengfeiyue/svc_dataset/base_210 --f0_predictor dio 


# if [ ! -d "logs/${data_name}" ]; then
#     mkdir logs/${data_name}
# fi
# cp logs/base_768/* logs/${data_name}/
#cp configs/config_${data_name}.json logs/${data_name}/config.json
CUDA_VISIBLE_DEVICES=${ngpu} python train.py -c configs/config_${data_name}.json -m ${data_name} -p ${port}
