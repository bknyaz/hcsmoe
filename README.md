# Fork of HC-SMoE for Qwen3 MoE models

[![arXiv](https://img.shields.io/badge/arXiv-2503.10522-brightgreen.svg?style=flat-square)](https://arxiv.org/abs/2410.08589)

- Original code (HC-SMoE): https://github.com/wazenmai/HC-SMoE
- Baseline (MC-SMoE): https://github.com/UNITES-Lab/MC-SMoE

## Setup

Install the following recommended versions:

```
pip install langdetect immutabledict torch_geometric bitsandbytes torch==2.7.1 torchvision torchtext torchaudio transformers==4.55.4 datasets vllm==0.10.1.1 numpy==1.26.4;
pip install evaluate==0.4.6  # use fresh version of evaluate
pip install fire deepspeed

cd $HOME;
git clone git@github.com:EleutherAI/lm-evaluation-harness.git
cd lm-evaluation-harness
pip install -e .
cd $HOME;
```

## Run merging and small tasks eval

For detailed description for each argument, please see [here](./scripts/README.md).
```
# adjust the paths and other arguments in the following script before running it
bash scripts/qwen/run.sh
```

## Evaluation on larger tasks

Example for IFeval using 4xA100/H100 GPUs:
```
python -m lm_eval --model vllm --model_args pretrained=Qwen/Qwen3-30B-A3B-Instruct-2507,tensor_parallel_size=4,dtype=auto,gpu_memory_utilization=0.9,data_parallel_size=1 --tasks ifeval --batch_size 1 --apply_chat_template=True --confirm_run_unsafe_code
```
tip: in case of OOM use max_model_len=131072 (or smaller number).
