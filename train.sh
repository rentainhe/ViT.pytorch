set -aux

NAME="Baseline"
GPUS="0,1"

# DDP settings
# Use IFS to spliting variables for counting nproc_per_node
IFS=','
GPUs=($GPU)
NPROC_PER_NODE=${#GPUs[@]}
IFS=''
MASTER_PORT=12345

python -m torch.distributed.launch --nproc_per_node=$NPROC_PER_NODE --master_port $MASTER_PORT debug_train.py --name $NAMEs