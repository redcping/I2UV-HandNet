
python super_mesh.py --data_dir ${DATA_DIR} \
                     --output_dir ${OUTPUT_DIR} \
                     --model_dir ${MODEL_DIR} \
                     --load_path=xvx_frei_130k_hand.pth \
                     --datanames=uvz_778_3k_pair \
                     --batch_size=128  \
                     --init_lr=0.0001 \
                     --img_size=256 \
                     --encoder_pretrained=True #\
                     #--do_score=True #\
                     #--do_sat=True
