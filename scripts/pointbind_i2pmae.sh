CKPT=./ckpts/pointbind_i2pmae.pt
MODEL=PointBind_I2PMAE

OUTPUT_DIR=./outputs/test_pointbind_8kpts

CUDA_VISIBLE_DEVICES=0 python zero_shot.py \
--model $MODEL \
--npoints 8192 \
--output-dir $OUTPUT_DIR \
--ckpt_path $CKPT \
2>&1 | tee $OUTPUT_DIR/log.txt
