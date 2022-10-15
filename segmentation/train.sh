###### !/usr/bin/env sh

CWD=`pwd` 
HDFS=hdfs://hobot-bigdata/
#set this to enable reading from hdfs
export CLASSPATH=$HADOOP_PREFIX/lib/classpath_hdfs.jar
export JAVA_TOOL_OPTIONS="-Xms512m -Xmx10000m"

cd ${WORKING_PATH}
cp -r ${WORKING_PATH}/* /job_data


# CONDA_ENV_NAME=conda_env_mm3d_8

# ln -s /cluster_home/custom_data/nuscenes  ${WORjKING_PATH}/data/


####get gcc-5.5.0
echo get gcc-5.5.0
hdfs dfs -get hdfs://hobot-bigdata/user/tianheng.cheng/envs/gcc.tar && tar xf gcc.tar
export PATH=${WORKING_PATH}/gcc/bin/:$PATH
export LD_LIBRARY_PATH=${WORKING_PATH}/gcc/lib:${WORKING_PATH}/gcc/lib64:$LD_LIBRARY_PATH

hdfs dfs -get hdfs://hobot-bigdata/user/tianheng.cheng/envs/3d.tar.gz && tar xf 3d.tar.gz

echo get 3d env
export PATH=${WORKING_PATH}/3d/bin/:$PATH



python setup.py build develop


DATASET=/cluster_home/custom_data/nuscenes
LABEL=/cluster_home/custom_data/nuscenes/cvt_labels_nuscenes


CONFIG=cvt_nuscenes_vehicle
# CONFIG=cvt_nuscenes_vehicle_k5x5
CONFIG=cvt_nuscenes_vehicle_k7x3_fixed
CONFIG=cvt_nuscenes_vehicle_3x3_wopos
CONFIG=cvt_nuscenes_vehicle_grid_3x3.yaml
CONFIG=cvt_nuscenes_vehicle_grid_7x3
CONFIG=cvt_nuscenes_vehicle_3x3_wopos_wofix
CONFIG=cvt_nuscenes_vehicle_index_3x3
CONFIG=cvt_nuscenes_vehicle_index_3x3_wopos
CONFIG=cvt_nuscenes_vehicle_index_3x3_wopos
CONFIG=cvt_nuscenes_vehicle_index_7x3
# CONFIG=cvt_nuscenes_vehicle_index_3x3_wopos_fixmask
# CONFIG=cvt_nuscenes_vehicle_index_7x3_revert
CONFIG=cvt_nuscenes_vehicle_15k.yaml
CONFIG=cvt_nuscenes_vehicle_index_3x3_15k.yaml
CONFIG=cvt_nuscenes_vehicle_index_3x3_large.yaml
CONFIG=cvt_nuscenes_vehicle_index_7x1_conv.yaml
CONFIG=cvt_nuscenes_vehicle_index_7x1_conv_woemb
CONFIG=cvt_nuscenes_vehicle_index_7x1_conv_imgemb
CONFIG=cvt_nuscenes_vehicle_index_7x1_conv_ada_emb
CONFIG=cvt_nuscenes_vehicle_index_7x1_pool_imgemb
CONFIG=cvt_nuscenes_vehicle_index_5x1_conv_imgemb.yaml
# ONFIG=cvt_nuscenes_vehicle_index_7x1_conv5_imgemb

CONFIG=cvt_nuscenes_vehicle_15k.yaml
CONFIG=cvt_nuscenes_vehicle_index_3x3_15k
CONFIG=cvt_nuscenes_vehicle_index_3x3_20k.yaml


# obtain setting1 labels

hdfs dfs -get hdfs://hobot-bigdata/user/tianheng.cheng/data/cvt_labels_setting1.tar.gz && tar -xf cvt_labels_setting1.tar.gz


# LABEL=${WORKING_PATH}/cvt_labels_setting1


sleep 1m
CONFIG=cvt_nuscenes_vehicle_index_3x3_conv
echo start training ${CONFIG}
python scripts/train.py +experiment=${CONFIG}  data.dataset_dir=${DATASET} data.labels_dir=${LABEL}


sleep 1m
CONFIG=cvt_nuscenes_vehicle_index_3x3_conv
echo start training ${CONFIG}
python scripts/train.py +experiment=${CONFIG}  data.dataset_dir=${DATASET} data.labels_dir=${LABEL}

# sleep 1m
# CONFIG=cvt_nuscenes_vehicle_setting2
# echo start training ${CONFIG}
# python scripts/train.py +experiment=${CONFIG}  data.dataset_dir=${DATASET} data.labels_dir=${LABEL}

# sleep 1m
# CONFIG=cvt_nuscenes_vehicle_index_3x3_setting
# echo start training ${CONFIG}
# python scripts/train.py +experiment=${CONFIG}  data.dataset_dir=${DATASET} data.labels_dir=${LABEL}



# sleep 1m
# CONFIG=cvt_nuscenes_vehicle_index_7x1_conv_imgemb_setting1
# echo start training ${CONFIG}
# python scripts/train.py +experiment=${CONFIG}  data.dataset_dir=${DATASET} data.labels_dir=${LABEL}

# sleep 1m
# CONFIG=cvt_nuscenes_vehicle_index_7x1_conv_imgemb_setting1
# echo start training ${CONFIG}
# python scripts/train.py +experiment=${CONFIG}  data.dataset_dir=${DATASET} data.labels_dir=${LABEL} model.encoder.bev_height=0.0





# # 1epoch
# sleep 1m
# MAX_ITERS=1800
# echo start training ${MAX_ITERS}
# python scripts/train.py +experiment=${CONFIG}  data.dataset_dir=${DATASET} data.labels_dir=${LABEL} trainer.max_steps=${MAX_ITERS}
# # 2epoch
# sleep 1m
# MAX_ITERS=3550
# echo start training ${MAX_ITERS}
# python scripts/train.py +experiment=${CONFIG}  data.dataset_dir=${DATASET} data.labels_dir=${LABEL} trainer.max_steps=${MAX_ITERS}
# # 3epoch
# sleep 1m
# MAX_ITERS=5300
# echo start training ${MAX_ITERS}
# python scripts/train.py +experiment=${CONFIG}  data.dataset_dir=${DATASET} data.labels_dir=${LABEL} trainer.max_steps=${MAX_ITERS}
# # 4epoch
# sleep 1m
# MAX_ITERS=7050
# echo start training ${MAX_ITERS}
# python scripts/train.py +experiment=${CONFIG}  data.dataset_dir=${DATASET} data.labels_dir=${LABEL} trainer.max_steps=${MAX_ITERS}
# # 5epoch
# sleep 1m
# MAX_ITERS=8800
# echo start training ${MAX_ITERS}
# python scripts/train.py +experiment=${CONFIG}  data.dataset_dir=${DATASET} data.labels_dir=${LABEL} trainer.max_steps=${MAX_ITERS}
# sleep 1m
# MAX_ITERS=10600
# echo start training ${MAX_ITERS}
# python scripts/train.py +experiment=${CONFIG}  data.dataset_dir=${DATASET} data.labels_dir=${LABEL} trainer.max_steps=${MAX_ITERS}



# CONFIG=cvt_nuscenes_vehicle_index_3x3_15k
# # 1epoch
# sleep 1m
# MAX_ITERS=1800
# echo start training ${MAX_ITERS}
# python scripts/train.py +experiment=${CONFIG}  data.dataset_dir=${DATASET} data.labels_dir=${LABEL} trainer.max_steps=${MAX_ITERS}
# # 2epoch
# sleep 1m
# MAX_ITERS=3550
# echo start training ${MAX_ITERS}
# python scripts/train.py +experiment=${CONFIG}  data.dataset_dir=${DATASET} data.labels_dir=${LABEL} trainer.max_steps=${MAX_ITERS}
# # 3epoch
# sleep 1m
# MAX_ITERS=5300
# echo start training ${MAX_ITERS}
# python scripts/train.py +experiment=${CONFIG}  data.dataset_dir=${DATASET} data.labels_dir=${LABEL} trainer.max_steps=${MAX_ITERS}
# # 4epoch
# sleep 1m
# MAX_ITERS=7050
# echo start training ${MAX_ITERS}
# python scripts/train.py +experiment=${CONFIG}  data.dataset_dir=${DATASET} data.labels_dir=${LABEL} trainer.max_steps=${MAX_ITERS}
# # 5epoch
# sleep 1m
# MAX_ITERS=8800
# echo start training ${MAX_ITERS}
# python scripts/train.py +experiment=${CONFIG}  data.dataset_dir=${DATASET} data.labels_dir=${LABEL} trainer.max_steps=${MAX_ITERS}
# sleep 1m
# MAX_ITERS=10600
# echo start training ${MAX_ITERS}
# python scripts/train.py +experiment=${CONFIG}  data.dataset_dir=${DATASET} data.labels_dir=${LABEL} trainer.max_steps=${MAX_ITERS}



# CONFIG=cvt_nuscenes_vehicle_index_7x1_conv_15k
# # 1epoch
# sleep 1m
# MAX_ITERS=1800
# echo start training ${MAX_ITERS}
# python scripts/train.py +experiment=${CONFIG}  data.dataset_dir=${DATASET} data.labels_dir=${LABEL} trainer.max_steps=${MAX_ITERS}
# # 2epoch
# sleep 1m
# MAX_ITERS=3550
# echo start training ${MAX_ITERS}
# python scripts/train.py +experiment=${CONFIG}  data.dataset_dir=${DATASET} data.labels_dir=${LABEL} trainer.max_steps=${MAX_ITERS}
# # 3epoch
# sleep 1m
# MAX_ITERS=5300
# echo start training ${MAX_ITERS}
# python scripts/train.py +experiment=${CONFIG}  data.dataset_dir=${DATASET} data.labels_dir=${LABEL} trainer.max_steps=${MAX_ITERS}
# # 4epoch
# sleep 1m
# MAX_ITERS=7050
# echo start training ${MAX_ITERS}
# python scripts/train.py +experiment=${CONFIG}  data.dataset_dir=${DATASET} data.labels_dir=${LABEL} trainer.max_steps=${MAX_ITERS}
# # 5epoch
# sleep 1m
# MAX_ITERS=8800
# echo start training ${MAX_ITERS}
# python scripts/train.py +experiment=${CONFIG}  data.dataset_dir=${DATASET} data.labels_dir=${LABEL} trainer.max_steps=${MAX_ITERS}

# sleep 1m
# MAX_ITERS=10600
# echo start training ${MAX_ITERS}
# python scripts/train.py +experiment=${CONFIG}  data.dataset_dir=${DATASET} data.labels_dir=${LABEL} trainer.max_steps=${MAX_ITERS}



# echo start training
# python scripts/train.py +experiment=${CONFIG}  data.dataset_dir=${DATASET} data.labels_dir=${LABEL}

# sleep 1m
# echo start training
# python scripts/train.py +experiment=${CONFIG}  data.dataset_dir=${DATASET} data.labels_dir=${LABEL}



# CONFIG=cvt_nuscenes_vehicle_index_7x1_conv_ada_emb

# echo start training
# python scripts/train.py +experiment=${CONFIG}  data.dataset_dir=${DATASET} data.labels_dir=${LABEL}


# CONFIG=cvt_nuscenes_vehicle_15k

# sleep 1m
# echo start training
# python scripts/train.py +experiment=${CONFIG}  data.dataset_dir=${DATASET} data.labels_dir=${LABEL}



# sleep 1m

# CONFIG=cvt_nuscenes_vehicle_index_3x3_10k

# echo start training
# python scripts/train.py +experiment=${CONFIG}  data.dataset_dir=${DATASET} data.labels_dir=${LABEL}

# sleep 1m

# echo start training
# python scripts/train.py +experiment=${CONFIG}  data.dataset_dir=${DATASET} data.labels_dir=${LABEL}


# sleep 1m

# CONFIG=cvt_nuscenes_vehicle_index_7x1_conv_ada_height1.0

# echo start training
# python scripts/train.py +experiment=${CONFIG}  data.dataset_dir=${DATASET} data.labels_dir=${LABEL}

# sleep 1m

# echo start training
# python scripts/train.py +experiment=${CONFIG}  data.dataset_dir=${DATASET} data.labels_dir=${LABEL}

# sleep 1m

# CONFIG=cvt_nuscenes_vehicle_index_7x1_conv_ada_height2.0

# echo start training
# python scripts/train.py +experiment=${CONFIG}  data.dataset_dir=${DATASET} data.labels_dir=${LABEL}

# sleep 1m

# echo start training
# python scripts/train.py +experiment=${CONFIG}  data.dataset_dir=${DATASET} data.labels_dir=${LABEL}




# sleep 30
# echo start training
# python scripts/train.py +experiment=${CONFIG}  data.dataset_dir=${DATASET} data.labels_dir=${LABEL}



# get ckpts
# echo get ckpts from hdfs
# mkdir ckpts
# cd ckpts
# hdfs dfs -get  hdfs://hobot-bigdata/user/shaoyu.chen/detr3d_ckpts/*
# cd ..

# ####install 
# echo insall mmcv mmdet mmseg
# pip3 install Pillow==6.2.2 --user
# MMCV_WITH_OPS=1 pip3 install mmcv-full==1.3.13 --user -f https://download.openmmlab.com/mmcv/dist/cu111/torch1.9.1/index.html --user
# pip3 install mmdet==2.19  --user #==2.14.0   2.19 
# pip3 install mmsegmentation==0.19 --user #==0.14.1    0.19



# echo insall mmdetection3d
# cd mmdetection3d
# rm build -r
# find . -name "*.so" | xargs rm
# pip3 install -r requirements.txt --user
# # export PATH=/home/users/shaoyu.chen/.local/bin:$PATH
# # python3 setup.py install --prefix=$(pwd)
# python3 setup.py install --user
# # export PYTHONPATH=$(pwd):$PYTHONPATH
# cd ..


# echo install deconvv2
# cd projects/mmdet3d_plugin/deconvv2_ops
# rm build -r
# find . -name "*.so" | xargs rm
# python3 setup.py install --user
# cd ../../..




# echo training


# chmod +x tools/dist_train.sh

# CONFIG=bev_detr3d_res50_gridmask.py

# tools/dist_train.sh projects/configs/detr3d/${CONFIG} 8 --work-dir /job_data/work_dirs


# wget http://fm-shaoyu-chen.ucloudtrain.hogpu.cc/plat_gpu/cluster_3090--try_v62_1_r50_fp16_p6-20220410-112850-COPY/output/work_dirs/epoch_18.pth
# tools/dist_train.sh projects/configs/detr3d/${CONFIG}  8 \
#     --work-dir /job_data/work_dirs  \
    # --resume-from ./epoch_20.pth


mkdir /job_data/output_results

mv outputs /job_data/output_results/
mv logs /job_data/output_results/

echo sleep 86
sleep 1d

# mkdir /job_data/output_results_bak

# mv outputs /job_data/output_results_bak/
# mv logs /job_data/output_results_bak/