# ColA: Collaborative Adaptation with Gradient Learning
[arXiv] This is an implementation of [ColA: Collaborative Adaptation with Gradient Learning](https://arxiv.org/abs/2404.13844)
- Illustration of the Fine-Tuning as a Service (FTaaS) system architecture.
<img src="/asset/ftaas.png">

## Requirements
See `requirements.txt`

## Instructions
 - Use `verify.py` to verify method and theory
 - Experimental control are configured in `config.yml`
 - Use `make.sh` to generate run script with `make.py`
 - Use `make.py` to generate exp script to `scripts`
 - Use `make_dataset.py` to prepare datasets (SAMsum has to be downloaded manually)
 - Use `process.py` to process exp results
 - Experimental setup are listed in `make.py` 
 - Hyperparameters can be found at `process_control()` in utils.py 
 
## Examples
 - Train full fine-tuning for CoLA dataset (RoBERT (base), Sequence Classification, $B=32$)
    ```ruby
    python train_model.py --control_name glue-cola_roberta-base_sc_full_32
    ```
 - Train LoRA for FPB dataset (BART (base), Sequece to Sequence, $B=32$)
    ```ruby
    python train_peft.py --control_name fpb-sa_bart-base_s2s_lora_32
    ```
 - Train ColA (Low Rank) for WikiSQL dataset (BART (base), Sequece to Sequence, $I=1$, $B=32$)
    ```ruby
    python train_cola.py --init_seed 0 --world_size 1 --num_experiment 1 --resume_mode 1 --control_name wikisql_bart-base_s2s_cola-lowrank-1_32
    ```
 - Test ColA (Low Rank-Linear) for Dolly dataset (GPT-2, Causal Language Modeling, $I=1$, $B=32$, Collaboration)
    ```ruby
    python test_cola_dist.py --init_seed 0 --world_size 1 --num_experiment 1 --resume_mode 1 --control_name dolly-15k_gpt2_clm_cola-lowrank~linear-1_32_col
    ```

## Results
- Learning curves of (a) Linear (b) MLP and (c) CNN with the CIFAR10 dataset of IC task and Accuracy metric.
![ic_CIFAR10](/asset/ic_CIFAR10.png)

- Learning curves of (a) MNLI (b) SST-2, and (c) MRPC datasets of SC task with and GLUE metric.
![sc](/asset/sc.png)

- Learning curves of (a) GPT-2 and (b) Llama-2 (Q, V) on Dolly dataset of CLM task and ROUGE (Longest) metric.
![clm](/asset/clm.png)

## Acknowledgement
*Enmao Diao  
Qi Le  
Suya Wu  
Xinran Wang  
Ali Anwar  
Jie Ding  
Vahid Tarokh*
