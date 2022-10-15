###### !/usr/bin/env sh

CWD=`pwd` 
HDFS=hdfs://hobot-bigdata/
#set this to enable reading from hdfs
export CLASSPATH=$HADOOP_PREFIX/lib/classpath_hdfs.jar
export JAVA_TOOL_OPTIONS="-Xms512m -Xmx10000m"

cd ${WORKING_PATH}
cp -r ${WORKING_PATH}/* /job_data


# CONDA_ENV_NAME=conda_env_mm3d_8

# ln -s /cluster_home/custom_data/nuscenes  ${WORKING_PATH}/data/


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


python scripts/train.py +experiment=${CONFIG}  data.dataset_dir=${DATASET} data.labels_dir=${LABEL}



mkdir /job_data/output_results

mv outputs /job_data/output_results/
mv logs /job_data/output_results/
