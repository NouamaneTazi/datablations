import os
import subprocess

CHECKPOINT_PATH="Megatron-DeepSpeed/checkpoints/gpt2"

VOCAB_FILE="/project/project_462000119/nouatazi/data/gpt2/vocab.json"
MERGE_FILE="/project/project_462000119/nouatazi/data/gpt2/merges.txt"
DATA_PATH="/scratch/project_462000119/data/pile/megatron_data/meg-gpt2_pile_text_document"
TENSORBOARD_PATH="Megatron-DeepSpeed/output_dir/tensorboard"

def makejob(CHECKPOINT_PATH=CHECKPOINT_PATH, 
            VOCAB_FILE=VOCAB_FILE, 
            MERGE_FILE=MERGE_FILE, 
            DATA_PATH=DATA_PATH, 
            TENSORBOARD_PATH=TENSORBOARD_PATH,
            MICRO_BATCH_SIZE=1,
            GLOBAL_BATCH_SIZE=16,
            TP_SIZE=1,
            PP_SIZE=1,
            NLAYERS=24,
            NHIDDEN=2048,
            NHEADS=16,
            SEQ_LEN=2048,
            FFN_HIDDEN_SIZE=8192,
            # VOCAB_SIZE=50257,
            SAVE_INTERVAL=50,
            TRAIN_SAMPLES="10_000",
            NNODES=1,
            GPUS_PER_NODE=2,
            LR=2e-4,
            ):
    return f"""#!/bin/bash

#SBATCH --job-name=1B3-alibi.slurm
#SBATCH --nodes={NNODES}
#SBATCH --ntasks-per-node=1          # crucial - only 1 task per dist per node!
#SBATCH --cpus-per-task=40         # number of cores per tasks
#SBATCH --hint=nomultithread         # we get physical cores not logical
#SBATCH --gpus-per-node=mi250:{GPUS_PER_NODE}  # number of gpus per node
#SBATCH --time 12:00:00              # maximum execution time (HH:MM:SS)
#SBATCH -p pilot
#SBATCH --account=project_462000119
#SBATCH -o logs/tr7a-1B3-alibi.%j.out
#SBATCH -e logs/tr7a-1B3-alibi.%j.err

module --quiet purge
module load cray-python

source {os.environ['VIRTUAL_ENV']}/bin/activate

set -x -e

# shared folder
six_ALL_CCFRWORK=/scratch/project_462000119


# TODO: modify these for your training setup, just Ctrl-F replace <YOUR_TRAINING_NAME>
DATA_OUTPUT_PATH=/project/project_462000119/nouatazi/scaling_laws_experiments
CHECKPOINT_PATH=$DATA_OUTPUT_PATH/checkpoints
REPO_PATH=$DATA_OUTPUT_PATH/tr7a-1B3-alibi-logs
TENSORBOARD_PATH=$REPO_PATH/tensorboard
CODECARBON_PATH=$REPO_PATH/codecarbon
LOGS_PATH=$REPO_PATH/logs
MEGATRON_DEEPSPEED_REPO=/project/project_462000119/nouatazi/Megatron-DeepSpeed


# TODO: you may change the dataset, some examples are at tr3-1B3-baseline (tr3 = c4 + t5-tokenizer, tr3m = the Pile)
VOCAB_FILE=$MEGATRON_DEEPSPEED_REPO/data/gpt2/vocab.json
MERGE_FILE=$MEGATRON_DEEPSPEED_REPO/data/gpt2/merges.txt
DATA_PATH=$six_ALL_CCFRWORK/data/pile/megatron_data/meg-gpt2_pile_text_document

# defining the right environment variables
export TRANSFORMERS_CACHE=$six_ALL_CCFRWORK/models
export HF_DATASETS_CACHE=$six_ALL_CCFRWORK/datasets
export HF_MODULES_CACHE=$six_ALL_CCFRWORK/modules
export HF_METRICS_CACHE=$six_ALL_CCFRWORK/metrics
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
cd $MEGATRON_DEEPSPEED_REPO

# so processes know who to talk to
MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
MASTER_PORT=6000

# TODO: this is our base config for 1B3, edit PP/TP/batch size/model config if smaller or bigger
GPUS_PER_NODE={GPUS_PER_NODE}
NNODES={NNODES}
PP_SIZE={PP_SIZE} # NLAYERS must be a multiple of PP_SIZE here
TP_SIZE={TP_SIZE} # always fixed to the size of a single node
DP_SIZE=$((NNODES*GPUS_PER_NODE/(PP_SIZE*TP_SIZE))) # will get derived automatically by trainer

MICRO_BATCH_SIZE={MICRO_BATCH_SIZE}
GLOBAL_BATCH_SIZE={GLOBAL_BATCH_SIZE}
TRAIN_SAMPLES={TRAIN_SAMPLES}


NLAYERS={NLAYERS}
NHIDDEN={NHIDDEN}
NHEADS={NHEADS}
FFN_HIDDEN_SIZE={FFN_HIDDEN_SIZE}
SEQ_LEN={SEQ_LEN}

SAVE_INTERVAL={SAVE_INTERVAL}

OPTIMIZER_ARGS=" \
    --optimizer adam \
    --adam-beta1 0.9 \
    --adam-beta2 0.999 \
    --adam-eps 1e-8 \
    --lr {LR} \
    --min-lr 1e-5 \
    --lr-decay-style cosine \
    --lr-decay-samples 73_242_187 \
    --lr-warmup-samples 183_105 \
    --clip-grad 1.0 \
    --weight-decay 1e-1 \
    "

EXIT_OPTS=" \
    --exit-duration-in-mins 1190 \
    "

GPT_ARGS=" \
    --num-layers $NLAYERS \
    --hidden-size $NHIDDEN \
    --num-attention-heads $NHEADS \
    --ffn-hidden-size $FFN_HIDDEN_SIZE \
    --seq-length $SEQ_LEN \
    --max-position-embeddings $SEQ_LEN \
    --micro-batch-size $MICRO_BATCH_SIZE \
    --global-batch-size $GLOBAL_BATCH_SIZE \
    --rampup-batch-size 32 32 2_000_000 \
    --train-samples $TRAIN_SAMPLES \
    --vocab-file $VOCAB_FILE \
    --merge-file $MERGE_FILE \
    --loss-scale 12 \
    --clip-grad 1.0 \
    --fp16 \
    --checkpoint-activations \
    --position-embedding-type alibi \
    $OPTIMIZER_ARGS \
    $EXIT_OPTS \
    "

OUTPUT_ARGS=" \
    --log-interval 200 \
    --save-interval $SAVE_INTERVAL \
    --eval-interval 1000 \
    --eval-iters 100 \
    --tensorboard-dir $TENSORBOARD_PATH \
    --tensorboard-queue-size 5 \
    --log-timers-to-tensorboard \
    --log-batch-size-to-tensorboard \
    --log-validation-ppl-to-tensorboard \
    "
# TODO: Add --codecarbon-dir $CODECARBON_PATH \ if you want to use codecarbon, not adding it for now to make the current
# series of experiments consistent, especially speed-wise. Adding it once Tr6 and Tr7 are done

ZERO_STAGE=1

config_json="./ds_config.$SLURM_JOBID.json"

# Deepspeed figures out GAS dynamically from dynamic GBS via set_train_batch_size()
cat <<EOT > $config_json
{{
  "train_micro_batch_size_per_gpu": $MICRO_BATCH_SIZE,
  "train_batch_size": $GLOBAL_BATCH_SIZE,
  "gradient_clipping": 1.0,
  "zero_optimization": {{
    "stage": $ZERO_STAGE
  }},
  "fp16": {{
    "enabled": true,
    "loss_scale": 0,
    "loss_scale_window": 500,
    "hysteresis": 2,
    "min_loss_scale": 1,
    "initial_scale_power": 12
  }},
  "steps_per_print": 2000,
  "wall_clock_breakdown": false
}}
EOT


DEEPSPEED_ARGS=" \
    --deepspeed \
    --deepspeed_config $config_json \
    --zero-stage $ZERO_STAGE \
    --deepspeed-activation-checkpointing \
    "

export LAUNCHER="python -u -m torch.distributed.launch \
    --nproc_per_node $GPUS_PER_NODE \
    --nnodes $NNODES \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT \
    "

export CMD=" \
    `pwd`/pretrain_gpt.py \
    --tensor-model-parallel-size $TP_SIZE \
    --pipeline-model-parallel-size $PP_SIZE \
    $GPT_ARGS \
    $OUTPUT_ARGS \
    --save $CHECKPOINT_PATH \
    --load $CHECKPOINT_PATH \
    --data-path $DATA_PATH \
    --data-impl mmap \
    --split 949,50,1 \
    --distributed-backend nccl \
     $DEEPSPEED_ARGS \
    "

# # clear old checkpoint as it'd mismatch while we sort things out
#     rm -rf $SAVE_CHECKPOINT_PATH


echo $CMD

# to debug - add echo (it exits and prints what it would have launched)
srun --jobid $SLURM_JOBID bash -c '$LAUNCHER --node_rank $SLURM_PROCID $CMD' 2>&1 | tee -a $LOGS_PATH/tr7a-1B3-alibi.$SLURM_JOBID.out

"""

def submit_job(job):
    with open('job.sbatch', 'w') as fp:
        fp.write(job)
    os.system("sbatch job.sbatch")
    # you can check the file job.sbatch to see what is being submitted

# Ensure the log directory exists
os.system("mkdir -p logs")

# Launch the batch jobs
submit_job(makejob())

# View logs
# tail -f  logs/<JOB_ID>.out logs/<JOB_ID>.err
