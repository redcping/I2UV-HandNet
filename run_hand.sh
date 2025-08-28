#!/bin/sh
PWD=$(cd -P -- "$(dirname -- "$0")" && pwd -P)
ls -l $PWD
mkdir ~/.pip/
cat > ~/.pip/pip.conf << EOF
[global]
index-url = http://jfrog.cloud.qiyi.domain/api/pypi/pypi/simple
trusted-host = jfrog.cloud.qiyi.domain

extra-index-url = http://jfrog.cloud.qiyi.domain/api/pypi/iqiyi-pypi-mesos/simple
EOF
#pip uninstall tensorflow-gpu
#pip install tensorboard==2.2.1
pip install torch==1.2.0
pip install torchvision==0.4.0
pip install -r requirements.txt
#multi_scale
#--datanames=hand_HUMBCP,syth,hand_HUMBCP2,cmu, xvx_frei_130k, youtube_linear_interpolation, obman,freihand_eval

if [ ${PREVIOUS_JOB_OUTPUT_DIR} ]; then
   ls -l $PREVIOUS_JOB_OUTPUT_DIR
   python youtube_uvz_train.py --data_dir ${DATA_DIR} \
                       --output_dir ${OUTPUT_DIR} \
                       --model_dir ${MODEL_DIR} \
                       --previous_job_output_dir ${PREVIOUS_JOB_OUTPUT_DIR} \
                       --load_path=hand_recon_parameters_epoch27.pth \
                       --encoder_name=resnet50 \
                       --do_uvz=True \
                       --datanames=xvx_frei_130k \
                       --batch_size=128 \
                       --init_lr=0.0001 \
                       --img_size=256 
                       #--do_multi_scale=True
else
   python youtube_uvz_train.py --data_dir ${DATA_DIR} \
                       --output_dir ${OUTPUT_DIR} \
                       --model_dir ${MODEL_DIR} \
                       --encoder_name=resnet50 \
                       --load_path=resnet50.pth \
                       --datanames=xvx_frei_130k,obman,youtube \
                       --batch_size=256  \
                       --init_lr=0.0001 \
                       --img_size=256 \
                       --use_multiple_gpu=True \
                       --encoder_pretrained=True \
                       --do_mano_cascade=True #\
                       #--do_sat=True
fi
    #python uvz_train.py --data_dir ${DATA_DIR} --output_dir ${OUTPUT_DIR} --model_dir ${MODEL_DIR} --do_uvz=True --datanames=xvx_frei_130k,syth,hand_HUMBCP2,cmu,hand_HUMBCP --batch_size=32  --init_lr=0.0001 --img_size=256 --encoder_pretrained=False --load_path=xvx_frei_130k_hand.pth
