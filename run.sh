#!/bin/bash

export CUDA_DEVICE_MAX_CONNECTIONS=1

GPUS_PER_NODE=4
# Change for multinode config
MASTER_ADDR=localhost
MASTER_PORT=6000
NNODES=1
NODE_RANK=0
export WORLD_SIZE=$(($GPUS_PER_NODE*$NNODES))

DISTRIBUTED_ARGS="
    --nproc_per_node $GPUS_PER_NODE \
    --nnodes $NNODES \
    --node_rank $NODE_RANK \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT
"
# [8192, 16, 128]
python benchmark.py --kernel flash --batch_size 1 --seq_length 8192 --num_head 16 --head_dim 128
torchrun $DISTRIBUTED_ARGS benchmark.py --kernel dist --batch_size 1 --seq_length 8192 --num_head 16 --head_dim 128

# [8192, 8, 128]
python benchmark.py --kernel flash --batch_size 1 --seq_length 8192 --num_head 8 --head_dim 128
torchrun $DISTRIBUTED_ARGS benchmark.py --kernel dist --batch_size 1 --seq_length 8192 --num_head 8 --head_dim 128

# [8192, 4, 128]
python benchmark.py --kernel flash --batch_size 1 --seq_length 8192 --num_head 4 --head_dim 128
torchrun $DISTRIBUTED_ARGS benchmark.py --kernel dist --batch_size 1 --seq_length 8192 --num_head 4 --head_dim 128

# [16384, 16, 128]
python benchmark.py --kernel flash --batch_size 1 --seq_length 16384 --num_head 16 --head_dim 128
torchrun $DISTRIBUTED_ARGS benchmark.py --kernel dist --batch_size 1 --seq_length 16384 --num_head 16 --head_dim 128

# [16384, 8, 128]
python benchmark.py --kernel flash --batch_size 1 --seq_length 16384 --num_head 8 --head_dim 128
torchrun $DISTRIBUTED_ARGS benchmark.py --kernel dist --batch_size 1 --seq_length 16384 --num_head 8 --head_dim 128

# [16384, 4, 128]
python benchmark.py --kernel flash --batch_size 1 --seq_length 16384 --num_head 4 --head_dim 128
torchrun $DISTRIBUTED_ARGS benchmark.py --kernel dist --batch_size 1 --seq_length 16384 --num_head 4 --head_dim 128

# [32768, 16, 128]
python benchmark.py --kernel flash --batch_size 1 --seq_length 32768 --num_head 16 --head_dim 128
torchrun $DISTRIBUTED_ARGS benchmark.py --kernel dist --batch_size 1 --seq_length 32768 --num_head 16 --head_dim 128

# [32768, 8, 128]
python benchmark.py --kernel flash --batch_size 1 --seq_length 32768 --num_head 8 --head_dim 128
torchrun $DISTRIBUTED_ARGS benchmark.py --kernel dist --batch_size 1 --seq_length 32768 --num_head 8 --head_dim 128

# [32768, 4, 128]
python benchmark.py --kernel flash --batch_size 1 --seq_length 32768 --num_head 4 --head_dim 128
torchrun $DISTRIBUTED_ARGS benchmark.py --kernel dist --batch_size 1 --seq_length 32768 --num_head 4 --head_dim 128

# [65536, 16, 128]
python benchmark.py --kernel flash --batch_size 1 --seq_length 65536 --num_head 16 --head_dim 128
torchrun $DISTRIBUTED_ARGS benchmark.py --kernel dist --batch_size 1 --seq_length 65536 --num_head 16 --head_dim 128

# [65536, 8, 128]
python benchmark.py --kernel flash --batch_size 1 --seq_length 65536 --num_head 8 --head_dim 128
torchrun $DISTRIBUTED_ARGS benchmark.py --kernel dist --batch_size 1 --seq_length 65536 --num_head 8 --head_dim 128

# [65536, 4, 128]
python benchmark.py --kernel flash --batch_size 1 --seq_length 65536 --num_head 4 --head_dim 128
torchrun $DISTRIBUTED_ARGS benchmark.py --kernel dist --batch_size 1 --seq_length 65536 --num_head 4 --head_dim 128

# [131072, 16, 128]
python benchmark.py --kernel flash --batch_size 1 --seq_length 131072 --num_head 16 --head_dim 128
torchrun $DISTRIBUTED_ARGS benchmark.py --kernel dist --batch_size 1 --seq_length 131072 --num_head 16 --head_dim 128

# [131072, 8, 128]
python benchmark.py --kernel flash --batch_size 1 --seq_length 131072 --num_head 8 --head_dim 128
torchrun $DISTRIBUTED_ARGS benchmark.py --kernel dist --batch_size 1 --seq_length 131072 --num_head 8 --head_dim 128

# [131072, 4, 128]
python benchmark.py --kernel flash --batch_size 1 --seq_length 131072 --num_head 4 --head_dim 128
torchrun $DISTRIBUTED_ARGS benchmark.py --kernel dist --batch_size 1 --seq_length 131072 --num_head 4 --head_dim 128
