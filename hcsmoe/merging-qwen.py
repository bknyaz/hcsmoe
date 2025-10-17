# -*- coding: utf-8 -*-
# @Author: pingzhili
# @Time: 2024/2/18
import os
import gc
import sys
import time
import pickle
from typing import Optional

import numpy as np
import logging
import torch
from fire import Fire
from transformers import Qwen2MoeForCausalLM, AutoTokenizer, AutoModelForCausalLM

#from torch.distributed.tensor import DTensor
#DTensor.disable_tensor_parallel()  # or config with mesh=None

from hcsmoe.evaluation import get_minipile_dataloder, evaluate_minipile_perplexity, evaluate_fewshot, get_calib_dataloder
from hcsmoe.merging.grouping_qwen import ExpertsGrouperForQwen2MoE, merge_by_groups_with_usage_weighted, merge_by_groups_within_and_across_models 
#, merge_by_feature_selection

logger = logging.getLogger(__name__)

###############################
#  Helper class and functions #
###############################
class Args:
    def __init__(
        self,
        task,
        num_average_groups: int,
        model_name: Optional[str] = "Qwen/Qwen1.5-MoE-A2.7B-Chat",
        dominant: Optional[str] = "knowledge",
        similarity_base: Optional[str] = "router-logits",
        merge: Optional[str] = "zipit",
        mode: Optional[str] = "normal",
        dataset: Optional[str] = "c4",
        n_sentences: Optional[int] = 32,
        train_batch_size: Optional[int] = 4,
        eval_batch_size: Optional[int] = 32,
        partition: Optional[int] = 1,
        start_layer: Optional[int] = 0,
        output_path: Optional[str] = None,
        result_path: Optional[str] = None,
        model_path: Optional[str] = None,
        group_limit: Optional[int] = 4,
        data_limit: Optional[int] = 50000,
        num_fewshot: Optional[int] = 0,
        try_oracle: Optional[bool] = False,
        random_start_center: Optional[bool] = False,
        weight: Optional[str] = None,
        cluster: Optional[str] = "kmeans",
        linkage: Optional[str] = "ward",
        hierarchical_stopping_metric: Optional[str] = "silhouette",
        overlap_metric: Optional[str] = "kl-divergence",
        dynamic_group: Optional[bool] = False,
    ):
        self.task = task
        self.num_average_groups = num_average_groups
        self.model_name = model_name
        self.dominant = dominant
        self.similarity_base = similarity_base
        self.merge = merge
        self.mode = mode
        self.dataset = dataset
        self.n_sentences = n_sentences
        self.train_batch_size = train_batch_size
        self.eval_batch_size = eval_batch_size
        self.partition = partition
        self.start_layer = start_layer
        self.output_path = output_path
        self.result_path = result_path
        self.model_path = model_path
        self.group_limit = group_limit
        self.data_limit = data_limit
        self.num_fewshot = num_fewshot
        self.try_oracle = try_oracle
        self.random_start_center = random_start_center
        self.weight = weight
        self.cluster = cluster
        self.linkage = linkage
        self.hierarchical_stopping_metric = hierarchical_stopping_metric
        self.overlap_metric = overlap_metric
        self.dynamic_group = dynamic_group

def get_dataloader(args, tokenizer):
    return get_calib_dataloder(
        dataset=args.dataset,
        tokenizer=tokenizer,
        max_block_size=2048,
        n_blocks_for_stat=args.n_sentences, # 32, 128
        batch_size=args.train_batch_size,
        num_workers=4,
    )

def get_grouper(args, config):
    return ExpertsGrouperForQwen2MoE(
                config=config,
                similarity_base=args.similarity_base,
                start_layer=args.start_layer,
                group_limit=args.group_limit,
                data_limit=args.data_limit,
                random_start_center=args.random_start_center,
                cluster=args.cluster,
                linkage=args.linkage,
                hierarchical_stopping_metric=args.hierarchical_stopping_metric,
                overlap_metric=args.overlap_metric,
                dynamic_group=args.dynamic_group,
            )

def evaluation(args, model, tokenizer):
    result_dir = args.result_path.split("/")[:-1]
    result_dir = "/".join(result_dir)
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)

    # if eval_ppl:
    #     evaluate_minipile_perplexity(
    #         model, tokenizer=tokenizer, batch_size=eval_batch_size, log=True
    #     )

    if isinstance(args.task, str):
        evaluate_fewshot(
            model, tokenizer=tokenizer, task=args.task, num_fewshot=args.num_fewshot, output_path=args.result_path, log=True
        )
    else:
        for i, t in enumerate(args.task):
            evaluate_fewshot(
                model, tokenizer=tokenizer, task=t, num_fewshot=args.num_fewshot, eval_batch_size=args.eval_batch_size, output_path=args.result_path, log=True
            )

def print_usage_frequency(usage_dict):
    for k in usage_dict:
        for num in usage_dict[k]:
            print(round(num.item(), 4), end=',')
        print()

def remove_experts(model):
    for layer in model.model.layers:
        moe_layer = layer.mlp
        if getattr(moe_layer, '_groups', None) is not None:
            experts_to_delete = []
            experts_to_keep = []
            for group in moe_layer._groups:
                experts_to_keep.append(group[0])
                experts_to_delete.extend(group[1:])  # keep the first expert in each group
            experts_to_delete.sort(reverse=True)
            experts_to_keep.sort()
            experts_to_keep = np.array(experts_to_keep)
            for idx in experts_to_delete:
                del moe_layer.experts[idx]
            # print(f'keep {len(moe_layer.experts)} experts', experts_to_keep, flush=True)
            moe_layer.num_experts = len(moe_layer.experts)
            moe_layer.gate.weight.data = moe_layer.gate.weight.data[experts_to_keep]
            moe_layer.gate.out_features = len(moe_layer.experts)
            del moe_layer._groups  # no need to save this anymore
        else:
            print('no _groups in layer')
    model.config.num_experts = moe_layer.num_experts

    return model


def run_hcsmoe(
        task: str,
        num_average_groups: int,
        model_name: Optional[str] = "Qwen/Qwen1.5-MoE-A2.7B-Chat",
        dominant: Optional[str] = "knowledge", # random, frequency, knowledge
        similarity_base: Optional[str] = "router-logits", # router-logits, weight, expert-output
        merge: Optional[str] = "zipit", # no, freq, zipit, update, fix-dom, unmerge,ix-dom-same
        mode: Optional[str] = "normal", # normal, activation-with-router-logits, input-weight, all
        dataset: Optional[str] = "c4", # c4 or all
        n_sentences: Optional[int] = 32,
        train_batch_size: Optional[int] = 4,
        eval_batch_size: Optional[int] = 32,
        partition: Optional[int] = 1,
        start_layer: Optional[int] = 0,
        output_path: Optional[str] = None,
        result_path: Optional[str] = None,
        model_path: Optional[str] = None,
        group_limit: Optional[int] = 4,
        data_limit: Optional[int] = 1000000,
        random_start_center: Optional[bool] = False,
        num_fewshot: Optional[int] = 0,
        cluster: Optional[str] = "kmeans",
        linkage: Optional[str] = "ward",
        hierarchical_stopping_metric: Optional[str] = "silhouette",
        ingredient: Optional[str] = "act", # act, weight, act+weight
        overlap_metric: Optional[str] = "cosine", # kl-divergence, wasserstein, cosine
        dynamic_group: Optional[bool] = False,
):
    print(f"Merge model {model_name} with {num_average_groups} group, {dominant} dominant + {similarity_base} grouping + {merge} merge - {mode}, ingredient: {ingredient}, evaluate on {task}")
    print(f"Cluster: {cluster}, linkage: {linkage}, hierarchical_stopping_metric: {hierarchical_stopping_metric}, overlap_metric: {overlap_metric}, dynamic_group: {dynamic_group}")
    
    ### 1. Initialization
    args = Args(
        task=task,
        num_average_groups=num_average_groups,
        model_name=model_name,
        dominant=dominant,
        similarity_base=similarity_base,
        merge=merge,
        mode=mode,
        dataset=dataset,
        n_sentences=n_sentences,
        train_batch_size=train_batch_size,
        eval_batch_size=eval_batch_size,
        partition=partition,
        start_layer=start_layer,
        output_path=output_path,
        result_path=result_path,
        model_path=model_path,
        group_limit=group_limit,
        data_limit=data_limit,
        num_fewshot=num_fewshot,
        random_start_center=random_start_center,
        cluster=cluster,
        linkage=linkage,
        hierarchical_stopping_metric=hierarchical_stopping_metric,
        overlap_metric=overlap_metric,
        dynamic_group=dynamic_group,
    )
    torch.manual_seed(0)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    tokenizer.pad_token_id = tokenizer.eos_token_id
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype="auto", #torch.bfloat16,
        device_map="auto",
        cache_dir=os.environ['HF_HOME']
    )
    if model_path:
        model.load_state_dict(torch.load(model_name))
    model.eval()
    print(model)
    print("[HC-SMoE] Number of parameters before merging:", model.num_parameters(), sum([p.numel() for p in model.parameters()]))

    dataloader_for_merging = get_dataloader(args, tokenizer)
    grouper = get_grouper(args, model.config)

    # HC-SMoE!
    print("[HC-SMoE] Number of parameters before merging:", model.num_parameters())
    print(f"[HC-SMoE] Merging into average {num_average_groups} groups...")
    group_st = time.time()
    if merge in ["freq", "freq-align"] or dominant == "frequency":
        grouper.compute_all_usages(model, dataloader_for_merging)
        print_usage_frequency(grouper._usage_frequency_state_dict)
    if dynamic_group:
        grouper.compute_all_usages(model, dataloader_for_merging, mode=hierarchical_stopping_metric)
        print_usage_frequency(grouper._usage_frequency_state_dict)
    

    ### 2. Get dominant experts
    dom_experts = None
    if dominant == "random":
        grouper.group_experts_randomly(num_groups=num_average_groups)
        dom_experts = None
    elif dominant == "frequency":
        if similarity_base != "no":
            grouper.compute_all_similarities(model, dataloader_for_merging)
        dom_experts = grouper.group_experts_globally_from_dominant_experts(
            num_average_groups=num_average_groups, merging_layers=list(range(start_layer, model.config.num_hidden_layers))
        )
    elif dominant == "routing-score":
        grouper.compute_all_usages(model, dataloader_for_merging, mode="routing-score")
        print_usage_frequency(grouper._usage_frequency_state_dict)
        dom_experts = grouper.group_experts_globally_from_dominant_experts(
            num_average_groups=num_average_groups, merging_layers=list(range(start_layer, model.config.num_hidden_layers))
        )
    elif dominant == "no":
        if merge == 'no':
            dom_experts = None
        else:
            ### Clustering
            dom_experts = grouper.cluster_experts(model=model, dataloader=dataloader_for_merging, num_groups=num_average_groups)
    else:
        raise ValueError(f"Unknown dominant: {dominant}")   

    ### 3. Merging
    if merge in ["freq", "freq-align"]:
        model = merge_by_groups_with_usage_weighted(
            model, grouper=grouper, merging_layers=list(range(start_layer, model.config.num_hidden_layers)),
            align="align" in merge
        )
    elif merge != 'no':
        model = merge_by_groups_within_and_across_models(
            qwen_model=model,
            grouper=grouper,
            dataloader=dataloader_for_merging,
            merge=merge,
            mode=mode,
            partition=partition,
            core_experts=dom_experts,
            dominant_alone=False,
            usage_weighted=False,
            ingredient=ingredient,
        )
        
    print(f"[HC-SMoE] Merging time: {time.time() - group_st:.2f} seconds")

    ### 4. Grouping results
    print(f"[HC-SMoE] ========= Grouping results ========= ")
    for name, state in grouper.group_state_dict().items():
        if dom_experts is None:
            print(f"Group {name}: {state.tolist()}")
        else:
            print(f"Group {name}: {state.tolist()} (DOMs are {dom_experts[name]}, {len(dom_experts[name])})")

    del grouper

    model = remove_experts(model)

    ### 5. Save model
    print("[HC-SMoE] Number of parameters after merging:", model.num_parameters(), sum([p.numel() for p in model.parameters()]))
    if not os.path.exists(output_path+"/model"):
        os.makedirs(output_path+"/model")
    model.save_pretrained(output_path+"/model",
                          safe_serialization=True,
                          max_shard_size='10GB' if '-FP8' in args.model_name else '4GB')
    tokenizer.save_pretrained(output_path+"/model")
    print('saving done!')

    ### 6. Evaluation
    evaluation(args, model, tokenizer)


if __name__ == "__main__":
    Fire(run_hcsmoe)
