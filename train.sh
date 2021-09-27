set -aux

DATASET="cifar100"
MODEL_TYPE="ViT-B_16"
IMG_SIZE=224
NAME="Test_Resume"
GPUS="0,1"
TRAIN_BATCH_SIZE=8
EVAL_BATCH_SIZE=8
GRAD_STEPS=2
NUM_STEPS=500
WARMUP_STEPS=100
DECAY_TYPE="cosine"
RESUME_PATH="/home/rentianhe/code/ViT-pytorch/output/Baseline/steps=100_checkpoint.ckpt"
# PRETRAINED_DIR=""



# DDP settings
# Use IFS to spliting variables for counting nproc_per_node
IFS=','
USED_GPU=($GPUS)
NPROC_PER_NODE=${#USED_GPU[@]}
IFS=''
MASTER_PORT=12345

python -m torch.distributed.launch --nproc_per_node=$NPROC_PER_NODE --master_port $MASTER_PORT train.py --name $NAME \
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
                                                                                                           --resume_path $RESUME_PATH \
                                                                                                        #    --pretrained_dir $PRETRAINED_DIR \
                                                                                                              