set -aux

DATASET="cifar100"
MODEL_TYPE="ViT-B_16"
IMG_SIZE=224
NAME="default_fp16"
GPUS="0,1"
TRAIN_BATCH_SIZE=512
EVAL_BATCH_SIZE=8
GRAD_STEPS=128
NUM_STEPS=500
WARMUP_STEPS=100
DECAY_TYPE="cosine"
FP16_OPT_LEVEL="O2"

# Note that if you set RESUME_PATH, the PRETRAINED_DIR will be deprecated
RESUME_PATH=""
PRETRAINED_DIR="checkpoint/ViT-B_16.npz"



# DDP settings
# Use IFS to spliting variables for counting nproc_per_node
IFS=','
USED_GPU=($GPUS)
NPROC_PER_NODE=${#USED_GPU[@]}
IFS=''
MASTER_PORT=12345

python -m torch.distributed.launch --nproc_per_node=$NPROC_PER_NODE --master_port $MASTER_PORT \
                                                                    train.py --name $NAME \
                                                                             --gpu $GPUS \
                                                                             --train_batch_size $TRAIN_BATCH_SIZE \
                                                                             --eval_batch_size $EVAL_BATCH_SIZE \
                                                                             --num_steps $NUM_STEPS \
                                                                             --decay_type $DECAY_TYPE \
                                                                             --warmup_steps $WARMUP_STEPS \
                                                                             --gradient_accumulation_steps $GRAD_STEPS \
                                                                             --model_type $MODEL_TYPE \
                                                                             --img_size $IMG_SIZE \
                                                                             --dataset $DATASET \
                                                                             --pretrained_dir $PRETRAINED_DIR \
                                                                             --fp16 \
                                                                             --fp16_opt_level $FP16_OPT_LEVEL
                                                                            #  --resume_path $RESUME_PATH \
                                                                                                                        