# Fork for benchmarking dist attention
## Notation
- `s` sequence length
- `b` (micro) batch size
- `h` hidden size
- `n` number of attention head
- `d` size of each attention head
## Principle
For an input activation of shape: `[s, b, h]`,  the QKV linear projects and splits it into three tensors of shape
`s, b, n, d`.

To enable sequence parallelism for attention, two dimensions can be taken into consideration.
Megatron-LM and Deepspeed partition at `n` and their computations are done by `FlashAttn`. By contrast, `DistAttn` of this repo, along with some other solutions partitions at `s`. To collect the results, some communication are necessary.
`DistAttn` is a neat solution partitioning at `s`, as it provides some optimization for the causal attention and typical overlapping.

This fork performs a **CLEAN** comparison against this two types of solutions.
Given an activation `[s, b, h]` and the degree of sequence parallelism `p`, I collect the duration of `FlashAttn` computing `[s, b, n/p, d]` and that of `DistAttn` computing `[s/p, b, n, d]`.

## Setup
DGX-A100

- pytorch   2.0.1 py3.9_cuda11.8_cudnn8.7.0_0  
- pytorch-cuda              11.8  
- triton                    2.1.0

## Result :sweat_smile:
```bash
bash run.sh
```

# Below is the original readme
# LightSeq: Sequence Level Parallelism for Distributed Training of Long Context Transformers
Official repository for LightSeq: Sequence Level Parallelism for Distributed Training of Long Context Transformers. LightSeq achieves up to 2x faster, 2-8x longer sequences vs Megatron-LM on 16 80GB A100s.

Paper: https://arxiv.org/pdf/2310.03294.pdf

### WARNING: This repo has not been fully tested for end-to-end training. We are actively working on that, with additional feature such as allowing padding in inputs. The current code is only for research preview and reproducing the paper results. We will make public announcement when more support are ready.

## News
- [2023/08] 🔥 Our paper is on! We provide a code preview of LightSeq. Stay tuned for future releases!

## LightSeq Implementation
* `lightseq_async_attn.py` contains codes for [DistAttn](https://github.com/RulinShao/LightSeq/blob/main/lightseq/lightseq_async_attn.py#L436) adapted from flash attention kernel.
* `async_communication.py` contains codes for communication-computation overlapped and workload-balanced communication.

## Usage
We provide an example to use lightseq in training. For example, to run LightSeq on 8 nodes, replace the `data_path` with your own dataset and run

```bash
python -m torch.distributed.run --nproc_per_node=8 \
         lightseq/train_lightseq_no_trainer.py \
        --model_name_or_path Llama-2-7b-chat-hf \
        --data_path <your_dataset>.pkl \
        --bf16 \
        --output_dir outputs \
        --num_train_epochs 3    \
        --per_device_train_batch_size 1 \
        --per_device_eval_batch_size 4  \
        --gradient_accumulation_steps 1 \
        --evaluation_strategy no \
        --save_strategy steps \
        --save_steps 1000  \
        --save_total_limit 1 \
        --learning_rate 2e-5 \
        --weight_decay 0.  \
        --warmup_ratio 0.03  \
        --lr_scheduler_type "cosine" \
        --logging_steps 1  \
        --fsdp "full_shard auto_wrap" \
        --fsdp_transformer_layer_cls_to_wrap 'LlamaDecoderLayer' \
        --tf32 True  \
        --model_max_length 16384  \
        --gradient_checkpointing True  \
        --lazy_preprocess True
```


## Rematerialization-aware gradient checkpointing
Rematerialization-aware gradient checkpointing can save your training time in one line! 
We release it as a Python package, [`fastckpt`](https://github.com/RulinShao/FastCkpt), so you install it by
```bash
pip install fastckpt
```

To replace **both** HF checkpointing with FashCkpt and HF LlamaAttention with FlashAttention, run 

```python
# import fastckpt before importing transformers
from fastckpt.llama_flash_attn_ckpt_monkey_patch import replace_hf_ckpt_with_fast_ckpt
replace_hf_ckpt_with_fast_ckpt()

# import transformers and other packages
import transformers
...
```

Alternatively, if you only want to replace the attention module with FlashAttention, simply run

```python
# import fastckpt before importing transformers
from fastckpt.llama_flash_attn_monkey_patch import replace_llama_attn_with_flash_attn
replace_llama_attn_with_flash_attn()

# import transformers and other packages
import transformers
...
```

If you find this repo useful, please cite
```
@article{li2023lightseq,
  title={LIGHTSEQ: SEQUENCE LEVEL PARALLELISM FOR DISTRIBUTED TRAINING OF LONG CONTEXT TRANS},
  author={Li, Dacheng and Shao, Rulin and Xie𝑠, Anze and Xing𝑐𝑚, Eric P and Gonzalez𝑏, Joseph E and Stoica𝑏, Ion and Ma𝑢, Xuezhe and Zhang𝑠, Hao},
  journal={arXiv preprint arXiv:2310.03294},
  year={2023}
}
```

