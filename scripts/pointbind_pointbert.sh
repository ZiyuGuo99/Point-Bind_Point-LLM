CKPT=./ckpts/pointbind_pointbert.pt
MODEL=PointBind_PointBERT

OUTPUT_DIR=./outputs/test_pointbind_8kpts

CUDA_VISIBLE_DEVICES=1 python zero_shot.py \
--model $MODEL \
--npoints 8192 \
--output-dir $OUTPUT_DIR \
--ckpt_path $CKPT \
2>&1 | tee $OUTPUT_DIR/log.txt
