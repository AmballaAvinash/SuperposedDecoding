#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 10 20:22:50 2024

@author: avinashamballa
"""

import json
import numpy as np
import os
import random
import torch
import datasets
from tqdm import tqdm
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.util import ngrams
from transformers import BitsAndBytesConfig
from transformers import LlamaTokenizer, LlamaForCausalLM, AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForCausalLM, GemmaForCausalLM, LogitsProcessorList, GPT2Tokenizer,GPT2LMHeadModel,T5ForConditionalGeneration, T5Tokenizer, MT5ForConditionalGeneration, M2M100ForConditionalGeneration, M2M100Tokenizer
import os
import pandas as pd
import json
import matplotlib.pyplot as plt
import time
from typing import List
import scipy.stats as stats 

import matplotlib
font = {'family' : 'normal',

        'weight' : 'bold',
        'size'   : 22}

matplotlib.rc('font', **font)


os.environ['KMP_DUPLICATE_LIB_OK']='True'

def load_hf_data_set(split,dataset_name, dataset_subname):
        data = {}
        data[split] = datasets.load_dataset(dataset_name,dataset_subname, split="validation",trust_remote_code=True, streaming=True)
        return data[split]


def ele_dist_k_from_idx(lst, start_index, k):
    return lst[start_index::k]



def run_experiments(model_name, data, batch, output_dir, shot, load_in_8bit=False):
    
    # model = "google/flan-t5-large"
    # tokenizer = AutoTokenizer.from_pretrained(model, use_fast=True)
    # model = AutoModelForSeq2SeqLM.from_pretrained(model).to('cuda')
   
    # # set pad_token_id to eos_token_id because GPT2 does not have a EOS token
    # model.config.pad_token_id = model.config.eos_token_id
    # model.generation_config.pad_token_id = model.config.eos_token_id
        
        
    # model = ["google/flan-t5-large" ,"meta-llama/Llama-2-7b-hf",  "google/gemma-2b-it", "google/gemma-7b-it"]

    cls_tokenizer = AutoTokenizer
    cls_model = AutoModelForCausalLM
    tokenizer_args = {}
    device_map="auto"
    torch_dtype=torch.float16
    
    if model_name=="llama-2":
        model = "meta-llama/Llama-2-7b-hf"
        cls_model = LlamaForCausalLM 
        cls_tokenizer = LlamaTokenizer
        tokenizer_args.update({"add_bos_token": True, "add_eos_token": False})
        
        # cls_tokenizer.pad_token = cls_tokenizer.eos_token
        # cls_model.config.pad_token_id = cls_model.config.eos_token_id
    elif model_name=="flan-t5":
        model = "google/flan-t5-large"
        cls_model = T5ForConditionalGeneration
        cls_tokenizer = T5Tokenizer
        # torch_dtype = torch.float32  # because of a logsoftmax error with half precision; TODO: double-check
    elif model_name=="gemma":
        model = "google/gemma-2b-it"
        cls_model = AutoModelForCausalLM
        cls_tokenizer = AutoTokenizer
        
    elif model_name == "mt5":
        model = "google/mt5-large"
        cls_model = MT5ForConditionalGeneration
        cls_tokenizer = T5Tokenizer
        
    elif model_name == "m2m":
        model = "facebook/m2m100_418M"
        cls_model = M2M100ForConditionalGeneration
        cls_tokenizer = M2M100Tokenizer
        
        
    elif model_name == "mt0":
        model = "bigscience/mt0-large"
        cls_model = AutoModelForSeq2SeqLM
        cls_tokenizer = AutoTokenizer
        
        
    tokenizer = cls_tokenizer.from_pretrained(model, **tokenizer_args)
    
    if  model_name == "m2m":
        tokenizer.src_lang = "fr"

    if load_in_8bit:
        # breakpoint()
        bnb_config= BitsAndBytesConfig(load_in_8bit=True,)
        model = cls_model.from_pretrained(model,
                                            torch_dtype=torch.bfloat16,
                                            device_map=device_map,
                                            quantization_config=bnb_config,
                                            # low_cpu_mem_usage=low_cpu_mem_usage,
                                            cache_dir = '/work/pi_dhruveshpate_umass_edu/aamballa_umass_edu/models/.cache',
                                            trust_remote_code=True,
                                            )
    
    else:
        model = cls_model.from_pretrained(model,
                                            torch_dtype=torch_dtype,
                                            device_map=device_map,
                                            # low_cpu_mem_usage=low_cpu_mem_usage,
                                            cache_dir = '/work/pi_dhruveshpate_umass_edu/aamballa_umass_edu/models/.cache',
                                            trust_remote_code=True,
                                            load_in_8bit=load_in_8bit)
    
    tokenizer.pad_token =  tokenizer.eos_token
    model.config.pad_token_id = model.config.eos_token_id
    model.eval()
    
        
    
    if shot=="oneshot": 
        default_fwd_instruction = "The French sentence: \"Ce serait le cas si des imitateurs de cette région s'assuraient l'exploitation des marques non protégées.\" translates to an English sentence: \"Particularly if imitators secure unprotected brand names there.\". Now, Translate the following French sentence to an English sentence."
    else:
        default_fwd_instruction = "Translate the following French sentence to an English sentence."
    default_fwd_input_prefix = "French sentence: "
    default_fwd_target_prefix = "English sentence: "
    
    results = {}
    for N in [20 , 10, 8, 5, 4, 2, 1]: # num_decodes (tp check ram issues)
        
        if N==40:
            trails = 1
        else:
            trails = 20   # 20
            
        # for temp in [0.1, 1]: # temperature
        #     for p in [0.8, 1.0]: # top p 
        #         for k in [30, 50]: # top k 
        #           for eps in [0.02, 0.05]: # epsilon
        
        for temp,p,k,eps in [[1,0.8,50,0.02]]: 
          
                    avg_max_bleu_arith_toppk = []
                    avg_max_bleu_temp_toppk = []
                    avg_max_bleu_greedy = []
                    avg_mbr_bleu_arith_toppk = []
                    avg_mbr_bleu_temp_toppk = []
                    avg_mbr_bleu_greedy = []
                    
                    avg_ngram_arith_toppk = []
                    avg_ngram_temp_toppk = []
                    avg_ngram_greedy = []
                    
                    avg_time_arith_toppk = []
                    avg_time_temp_toppk = []
                    avg_time_greedy = []
                    
                    
                    for i in range(trails):
                  
                         
                               
                           print(f"N {N}, Temp {temp}, Top p {p}, Top k {k}, Eps {eps}")
                           
                           output_dict = test(default_fwd_instruction, default_fwd_input_prefix, default_fwd_target_prefix, model_name, model, tokenizer, data, batch, output_dir, shot, N, temp, p, k , eps)
                          
                           # avg_max_bleu_arith_toppk = []
                           # avg_max_bleu_temp_toppk = []
                           # avg_max_bleu_greedy = []
                           # avg_mbr_bleu_arith_toppk = []
                           # avg_mbr_bleu_temp_toppk = []
                           # avg_mbr_bleu_greedy = []
                           
                           # avg_ngram_arith_toppk = []
                           # avg_ngram_temp_toppk = []
                           # avg_ngram_greedy = []
                           
                           # avg_time_arith_toppk = []
                           # avg_time_temp_toppk = []
                           # avg_time_greedy = []
                           
                           # for idx in range(len(data)):   
                           #     avg_max_bleu_arith_toppk.append(output_dict[idx]['max_bleu_score_arith_toppk'])
                           #     avg_max_bleu_temp_toppk.append(output_dict[idx]['max_bleu_score_temp_toppk'])
                           #     avg_max_bleu_greedy.append(output_dict[idx]['max_bleu_score_greedy'])
                               
                           #     avg_mbr_bleu_arith_toppk.append(output_dict[idx]['mbr_bleu_score_arith_toppk'])
                           #     avg_mbr_bleu_temp_toppk.append(output_dict[idx]['mbr_bleu_score_temp_toppk'])
                           #     avg_mbr_bleu_greedy.append(output_dict[idx]['mbr_bleu_score_greedy'])
                               
                           #     avg_ngram_arith_toppk.append(output_dict[idx]['n_gram_div_arith_toppk'])
                           #     avg_ngram_temp_toppk.append(output_dict[idx]['n_gram_div_temp_toppk'])
                               
                           #     avg_ngram_greedy.append(output_dict[idx]['n_gram_div_greedy'])
                               
                               
                           #     avg_time_arith_toppk.append(output_dict[idx]['time_arith_toppk'])
                           #     avg_time_temp_toppk.append(output_dict[idx]['time_temp_toppk'])
                               
                           #     avg_time_greedy.append(output_dict[idx]['time_greedy'])
                               
                           avg_max_bleu_arith_toppk.append(np.mean([item["max_bleu_score_arith_toppk"] for item in list(output_dict.values()) ], dtype="float64") )
                           avg_max_bleu_temp_toppk.append(np.mean([item["max_bleu_score_temp_toppk"] for item in list(output_dict.values()) ], dtype="float64"))
                           avg_max_bleu_greedy.append(np.mean([item["max_bleu_score_greedy"] for item in list(output_dict.values()) ], dtype="float64"))
                           
                           avg_mbr_bleu_arith_toppk.append(np.mean([item["mbr_bleu_score_arith_toppk"] for item in list(output_dict.values()) ], dtype="float64") )
                           avg_mbr_bleu_temp_toppk.append(np.mean([item["mbr_bleu_score_temp_toppk"] for item in list(output_dict.values()) ], dtype="float64") )
                           avg_mbr_bleu_greedy.append(np.mean([item["mbr_bleu_score_greedy"] for item in list(output_dict.values()) ], dtype="float64") )
                           
                           
                           avg_ngram_arith_toppk.append(np.mean([item["n_gram_div_arith_toppk"] for item in list(output_dict.values()) ], dtype="float64") )
                           avg_ngram_temp_toppk.append(np.mean([item["n_gram_div_temp_toppk"] for item in list(output_dict.values()) ], dtype="float64") )
        
                           avg_ngram_greedy.append(np.mean([item["n_gram_div_greedy"] for item in list(output_dict.values()) ], dtype="float64") )
                           
                           
                           avg_time_arith_toppk.append(np.mean([item["time_arith_toppk"] for item in list(output_dict.values()) ], dtype="float64") )
                           avg_time_temp_toppk.append(np.mean([item["time_temp_toppk"] for item in list(output_dict.values()) ], dtype="float64"))
                           avg_time_greedy.append(np.mean([item["time_greedy"] for item in list(output_dict.values()) ], dtype="float64"))
                           
                   
                    results = {}
                    results["Num Samples"] = len(data)
                    results["Num decodes"] = N
                    results["temperature"] = temp
                    results["Top p "] = p
                    results["Top k"] = k
                    results["Epsilon"] = eps
                    
                    results["Max BLEU Arithmetic Mean"] = np.mean(avg_max_bleu_arith_toppk)
                    results["Max BLEU Arithmetic Std"] = np.std(avg_max_bleu_arith_toppk)
                    results["Max BLEU Sampling Mean"] = np.mean(avg_max_bleu_temp_toppk)
                    results["Max BLEU Sampling Std"] = np.std(avg_max_bleu_temp_toppk)
                    results["Max BLEU Greedy Mean"] = np.mean(avg_max_bleu_greedy)
                    results["Max BLEU Greedy Std"] = np.std(avg_max_bleu_greedy)
                    
                    results["MBR BLEU Arithmetic Mean"] = np.mean(avg_mbr_bleu_arith_toppk)
                    results["MBR BLEU Arithmetic Std"] = np.std(avg_mbr_bleu_arith_toppk)
                    results["MBR BLEU Sampling Mean"] = np.mean(avg_mbr_bleu_temp_toppk)
                    results["MBR BLEU Sampling Std"] = np.std(avg_mbr_bleu_temp_toppk)
                    results["MBR BLEU Greedy Mean"] = np.mean(avg_mbr_bleu_greedy)
                    results["MBR BLEU Greedy Std"] = np.std(avg_mbr_bleu_greedy)
                    
                    results["N gram diversity Arithmetic Mean"] = np.mean(avg_ngram_arith_toppk)
                    results["N gram diversity Arithmetic Std"] = np.std(avg_ngram_arith_toppk)
                    results["N gram diversity Sampling Mean"] = np.mean(avg_ngram_temp_toppk)
                    results["N gram diversity Sampling Std"] = np.std(avg_ngram_temp_toppk)
                    results["N gram diversity Greedy Mean"] = np.mean(avg_ngram_greedy)
                    results["N gram diversity Greedy Std"] = np.std(avg_ngram_greedy)
                    
                    results["Time Arithmetic"] = avg_time_arith_toppk
                    results["Time Sampling"] = avg_time_temp_toppk
                    results["Time Greedy"] = avg_time_greedy
                    
                    with open(f'{output_dir}/{model_name}__{shot}__Samples_{len(data)}__N_{N}__temp_{temp}__p_{p}__k_{k}__eps_{eps}_results.json','w') as f:
                        json.dump(results,f)
                    
                   
                   

def test(default_fwd_instruction, default_fwd_input_prefix, default_fwd_target_prefix,  model_name, model, tokenizer, data, batch, output_dir, shot, N = 1, temp = 1, p=1, k = 50, eps = 0.02):
    
    num_return_sequences = 40
    
    if N==num_return_sequences:
            output_dict = {}
            
            for idx, d in enumerate(tqdm(data, desc="Predicting")):
                if idx%batch==0:
                    input_batch =[]
                    
                prompt_arr = [default_fwd_instruction,default_fwd_input_prefix]
                prompt_arr.append(d['fr'])
                prompt_arr.append(default_fwd_target_prefix)
                input_prompt = (' ').join(prompt_arr)  # join the sentences
                input_batch.append(input_prompt)
                
                output_dict[idx] = {}
                output_dict[idx]['gt'] = d['en']
                
                if (idx+1)%batch==0:
                    input_ids = tokenizer(input_batch, truncation=True, padding = True, return_tensors="pt").input_ids.to('cuda')
                    
                    if model_name == "m2m":
                        
                        
                        start_arith_toppk = time.time()
                        outputs_arith_toppk = model.generate(
                            input_ids = input_ids,
                            num_return_sequences = N,
                            do_sample = True,
                            temperature = temp,
                            top_p=p,
                            top_k=k,
                            num_beams = 1,
                            max_new_tokens = 100,
                            use_arithmetic = True, 
                            epsilon_cutoff = eps,
                            forced_bos_token_id=tokenizer.get_lang_id("en")
                            )
                        
                        end_arith_toppk  = time.time()
                        # print(end_arith_toppk - start_arith_toppk)
                        
                        
                        start_temp_toppk = time.time()
                        outputs_temp_toppk = model.generate(
                            input_ids = input_ids,
                            num_return_sequences = N,
                            do_sample = True,
                            temperature = temp,
                            top_p=p,
                            top_k=k,
                            num_beams = 1,
                            max_new_tokens = 100,
                            use_arithmetic = False, 
                            epsilon_cutoff = eps,
                            forced_bos_token_id=tokenizer.get_lang_id("en")
                            )
                        
                        end_temp_toppk = time.time()
                        # print(end_temp_toppk - start_temp_toppk)
                        
                        
                        start_greedy = time.time()
                        outputs_greedy = model.generate(
                            input_ids = input_ids,
                            num_return_sequences = 1,
                            forced_bos_token_id=tokenizer.get_lang_id("en")
                            )
                        end_greedy = time.time()
                        
                        
                    else:     
                        start_arith_toppk = time.time()
                        outputs_arith_toppk = model.generate(
                            input_ids = input_ids,
                            num_return_sequences = N,
                            do_sample = True,
                            temperature = temp,
                            top_p=p,
                            top_k=k,
                            num_beams = 1,
                            max_new_tokens = 100,
                            use_arithmetic = True,
                            epsilon_cutoff = eps
                            )
                        end_arith_toppk = time.time()
                        
                        start_temp_toppk = time.time()
                        outputs_temp_toppk = model.generate(
                            input_ids = input_ids,
                            num_return_sequences = N,
                            do_sample = True,
                            temperature = temp,
                            top_p=p,
                            top_k=k,
                            num_beams = 1,
                            max_new_tokens = 100,
                            use_arithmetic = False,
                            epsilon_cutoff = eps
                            )
                        end_temp_toppk = time.time()
                        
                        start_greedy = time.time()
                        outputs_greedy = model.generate(
                            input_ids = input_ids,
                            num_return_sequences = 1
                            )
                        end_greedy = time.time()
                          
                    
                    
                    decode_arith_toppk = [i.split('English sentence: ')[-1].strip('\n').replace("<pad>", "") for i in tokenizer.batch_decode(outputs_arith_toppk, skip_special_tokens=True)]  
                    decode_temp_toppk = [i.split('English sentence: ')[-1].strip('\n').replace("<pad>", "") for i in tokenizer.batch_decode(outputs_temp_toppk, skip_special_tokens=True)]   
                    decode_greedy = [i.split('English sentence: ')[-1].strip('\n').replace("<pad>", "") for i in tokenizer.batch_decode(outputs_greedy, skip_special_tokens=True)]   
        
        
                    num_decodes = 0
                    for j in range(idx-batch+1,idx+1):
                        output_dict[j]['arith_toppk'] = decode_arith_toppk[num_decodes:num_decodes+N]
                        output_dict[j]['temp_toppk'] = decode_temp_toppk[num_decodes:num_decodes+N]
                        output_dict[j]['greedy'] = [decode_greedy[j-(idx-batch+1)]]
                       
                        output_dict[j]['max_bleu_score_arith_toppk'], output_dict[j]['mbr_bleu_score_arith_toppk'], output_dict[j]['n_gram_div_arith_toppk'] = calculate_bleu_and_ngram_diversity(output_dict[j]['gt'], output_dict[j]['arith_toppk'])
                        output_dict[j]['max_bleu_score_temp_toppk'], output_dict[j]['mbr_bleu_score_temp_toppk'], output_dict[j]['n_gram_div_temp_toppk'] = calculate_bleu_and_ngram_diversity(output_dict[j]['gt'], output_dict[j]['temp_toppk'])
                        output_dict[j]['max_bleu_score_greedy'], output_dict[j]['mbr_bleu_score_greedy'], output_dict[j]['n_gram_div_greedy'] = calculate_bleu_and_ngram_diversity(output_dict[j]['gt'], output_dict[j]['greedy'])
        
        
                        output_dict[j]['time_arith_toppk'] = (end_arith_toppk- start_arith_toppk)/batch
                        output_dict[j]['time_temp_toppk'] = (end_temp_toppk - start_temp_toppk)/batch
                        output_dict[j]['time_greedy'] = (end_greedy - start_greedy)/batch
                        
                        
                        num_decodes+=N
                
             
            with open(f'{output_dir}/{model_name}__{shot}__Samples_{len(data)}__N_{N}__temp_{temp}__p_{p}__k_{k}__eps{eps}_output.json','w') as f:
                json.dump(output_dict,f, default=str)
                
            return output_dict
        
        
    else:
        
        
        with open(f'{output_dir}/{model_name}__{shot}__Samples_{len(data)}__N_{num_return_sequences}__temp_{temp}__p_{p}__k_{k}__eps{eps}_output.json') as f:
            output_dict = json.load(f)
            
        
        for idx, d in enumerate(tqdm(data, desc="Predicting")):
            idx = str(idx)
            offset = random.choice(range(num_return_sequences // N))
            output_dict[idx]['arith_toppk'] = ele_dist_k_from_idx(output_dict[idx]['arith_toppk'] , offset, num_return_sequences // N) 
    
            
            output_dict[idx]['temp_toppk'] = random.sample(output_dict[idx]['temp_toppk'], N) 
            output_dict[idx]['greedy'] = output_dict[idx]['greedy']
           
            output_dict[idx]['max_bleu_score_arith_toppk'], output_dict[idx]['mbr_bleu_score_arith_toppk'], output_dict[idx]['n_gram_div_arith_toppk'] = calculate_bleu_and_ngram_diversity(output_dict[idx]['gt'], output_dict[idx]['arith_toppk'])
            output_dict[idx]['max_bleu_score_temp_toppk'], output_dict[idx]['mbr_bleu_score_temp_toppk'], output_dict[idx]['n_gram_div_temp_toppk'] = calculate_bleu_and_ngram_diversity(output_dict[idx]['gt'], output_dict[idx]['temp_toppk'])
            output_dict[idx]['max_bleu_score_greedy'], output_dict[idx]['mbr_bleu_score_greedy'], output_dict[idx]['n_gram_div_greedy'] = calculate_bleu_and_ngram_diversity(output_dict[idx]['gt'], output_dict[idx]['greedy'])


            output_dict[idx]['time_arith_toppk'] = 0
            output_dict[idx]['time_temp_toppk'] = 0
            output_dict[idx]['time_greedy'] = 0
            
            
        return output_dict
            
        



def calculate_bleu_and_ngram_diversity(reference, translations):
    translations_split = [x.split() for x in translations]
    max_bleu_score = np.max([ np.round(100*sentence_bleu([reference.split()], x,  smoothing_function=SmoothingFunction().method4),3) for x in translations_split])

    translation_MBR = ""
    max_utility = 0
    for i in range(len(translations)):
        utility_score = 0
        for j in range(len(translations)):
            # utility_score += bleurt.score(translations[i], translations[j])   
            
            utility_score += sentence_bleu([translations[i].split()], translations[j].split(), smoothing_function=SmoothingFunction().method4)
            
        if max_utility < utility_score:
            max_utility = utility_score
            translation_MBR = translations[i]
                                  
    mbr_bleu_score = np.round(100*sentence_bleu([reference.split()], translation_MBR.split(),  smoothing_function=SmoothingFunction().method4),3) 


    n_values = [1, 2, 3, 4]  # BLUE-4 and ngram-4
    total_unique_ngrams = 0
    ngram_diversity_score = 0
    for n in n_values:
        unique_ngrams = set()
        total_ngram_count = 0
        for translation in translations:
            # Compute n-grams
            translation_ngrams = list(ngrams(translation.split(), n))
            # Count unique n-grams
            total_ngram_count += len(list(translation_ngrams))
            unique_ngrams.update(translation_ngrams)
        # Update total counts
        total_unique_ngrams = len(unique_ngrams)
        ngram_diversity_score += total_unique_ngrams / (total_ngram_count + torch.finfo(torch.float32).eps)
    return max_bleu_score, mbr_bleu_score,  ngram_diversity_score

if __name__ == "__main__":
    samplesize = 1000
    batch = 10
    output_dir = "outputs/WMT/V9"
    
    random.seed(41)
    # data =  random.sample(load_hf_data_set('validation','wmt14','fr-en')['translation'],samplesize)
    data = list(load_hf_data_set('validation','wmt14','fr-en').take(samplesize))
    data = [x["translation"] for x in data]

    
    # it versions and quantized
    for model_name in ["mt0"]:  #["llama-2", "flan-t5", "gemma", "mt5", "m2m", "mt0"]
        for shot in ["zeroshot"]:  # ["zeroshot", "oneshot"]
            print(f"{model_name}, {shot}")
            run_experiments(model_name, data, batch, output_dir, shot, load_in_8bit=False)
            
            
            # # tabular output resuls from json
            json_data = []
            for file in os.listdir(output_dir):
                if model_name in file and shot in file and "results" in file and str(samplesize)+"_" in file:
                    with open(os.path.join(output_dir,file),"r") as f:
                        jsondata = json.load(f)
                        json_data.append(jsondata)
            df = pd.DataFrame(json_data)
            df.to_csv(f"{output_dir}/Table_{model_name}__{shot}__Samples_{samplesize}.csv")
            
            # df = pd.read_csv(f"{output_dir}/Table_{model_name}__{shot}__Samples_{samplesize}.csv")
    
    
            # graphs
            fig1, axs1 = plt.subplots(4, 4, figsize=(12, 12))
            fig2, axs2 = plt.subplots(4, 4, figsize=(12, 12))
            fig3, axs3 = plt.subplots(4, 4, figsize=(12, 12))
            
            c = 0
            for temp in [0.1, 1]: # temperature
                for p in [0.8, 1.0]: # top p 
                   for k in [30, 50]: # top k 
                     for eps in [0.02, 0.05]: # epsilon
                     
                     
                      # p test 
                      
                      with open(f'{output_dir}/{model_name}__{shot}__Samples_{samplesize}__N_40__temp_{temp}__p_{p}__k_{k}__eps{eps}_output.json') as f:
                          output_dict = json.load(f)
                          
                      ptest_max_bleu_arith_toppk = []
                      ptest_max_bleu_temp_toppk = []
                      ptest_mbr_bleu_arith_toppk = []
                      ptest_mbr_bleu_temp_toppk = []                    
                     
                      for idx in range(samplesize):   
                         idx = str(idx)
                         ptest_max_bleu_arith_toppk.append(float(output_dict[idx]['max_bleu_score_arith_toppk']))
                         ptest_max_bleu_temp_toppk.append(float(output_dict[idx]['max_bleu_score_temp_toppk']))
                         
                         ptest_mbr_bleu_arith_toppk.append(float(output_dict[idx]['mbr_bleu_score_arith_toppk']))
                         ptest_mbr_bleu_temp_toppk.append(float(output_dict[idx]['mbr_bleu_score_temp_toppk']))
                         
                      p_test_max_bleu = stats.ttest_rel(ptest_max_bleu_arith_toppk, ptest_max_bleu_temp_toppk)
                      p_test_mbr_bleu = stats.ttest_rel(ptest_mbr_bleu_arith_toppk, ptest_mbr_bleu_temp_toppk)
                     
                      # print(f"Model: {model_name}, p-test Max BLEU {p_test_max_bleu.pvalue}, p-test MBR BLEU {p_test_mbr_bleu.pvalue} \n")
                      
                  
                    
                  
                      rows = df[(df["temperature"]==temp) & (df["Top p "]==p) & (df["Top k"]==k) & (df["Epsilon"]==eps)].sort_values("Num decodes")
                      x = rows["Num decodes"]
                      y_max_sample_bleu_mean = rows["Max BLEU Sampling Mean"] 
                      y_max_sample_bleu_std = rows["Max BLEU Sampling Std"] 
                      y_max_arit_bleu_mean = rows["Max BLEU Arithmetic Mean"]
                      y_max_arit_bleu_std = rows["Max BLEU Arithmetic Std"]
                      y_max_greedy_bleu_mean = rows["Max BLEU Greedy Mean"]
                      y_max_greedy_bleu_std = rows["Max BLEU Greedy Std"]
                      
                      y_mbr_sample_bleu_mean = rows["MBR BLEU Sampling Mean"] 
                      y_mbr_sample_bleu_std = rows["MBR BLEU Sampling Std"] 
                      y_mbr_arit_bleu_mean = rows["MBR BLEU Arithmetic Mean"]
                      y_mbr_arit_bleu_std = rows["MBR BLEU Arithmetic Std"]
                      y_mbr_greedy_bleu_mean = rows["MBR BLEU Greedy Mean"]
                      y_mbr_greedy_bleu_std = rows["MBR BLEU Greedy Std"]
                      
                      y_sample_div_mean = rows["N gram diversity Sampling Mean"] 
                      y_sample_div_std = rows["N gram diversity Sampling Std"] 
                      y_arit_div_mean = rows["N gram diversity Arithmetic Mean"] 
                      y_arit_div_std = rows["N gram diversity Arithmetic Std"] 
                      y_greedy_div_mean = rows["N gram diversity Greedy Mean"]
                      y_greedy_div_std = rows["N gram diversity Greedy Std"]
                      
                      y_sample_time = rows["Time Sampling"] 
                      y_arit_time = rows["Time Arithmetic"] 
                      y_greedy_time = rows["Time Greedy"]
                      
                      
                      i = int(c/4)
                      j = c%4
                      
                      axs1[i][j].set_xlabel("Num of sampled sequences")
                      axs1[i][j].set_ylabel("BLEU score")

                      axs1[i][j].errorbar(x,y_max_sample_bleu_mean, yerr=y_max_sample_bleu_std, fmt='o',label="Ancestral sampling (Max) ", color='darkorange', linestyle = "--")
                      axs1[i][j].errorbar(x,y_max_arit_bleu_mean, yerr=y_max_arit_bleu_std, fmt='^',label = "Arithemetic sampling (Max)", color='purple',  linestyle = "--")
                      
                      
                      axs1[i][j].errorbar(x,y_mbr_sample_bleu_mean, yerr=y_mbr_sample_bleu_std, fmt='o', label="Ancestral sampling (MBR) ", color='darkorange', linestyle = "-")
                      axs1[i][j].errorbar(x,y_mbr_arit_bleu_mean,yerr=y_mbr_arit_bleu_std, fmt='^', label = "Arithemetic sampling (MBR)", color='purple',  linestyle = "-")
                      # axs1[i][j].plot(x,y_mbr_greedy_bleu, marker='o',label = "Greedy (MBR)")
                      
                      axs1[i][j].errorbar(x,y_max_greedy_bleu_mean,yerr=y_max_greedy_bleu_std,  fmt='.', label = "Greedy" ,color='green',  linestyle = "-")
                      
                      
                      axs2[i][j].set_xlabel("Num of sampled sequences")
                      axs2[i][j].set_ylabel("N gram diversity")
                      axs2[i][j].errorbar(x,y_sample_div_mean, yerr=y_sample_div_std,  fmt='o',label="Ancestral sampling ", color='darkorange', linestyle = "-")
                      axs2[i][j].errorbar(x,y_arit_div_mean, yerr= y_arit_div_std, fmt='^',label = "Arithemetic sampling", color='purple', linestyle = "-")
                      axs2[i][j].errorbar(x,y_greedy_div_mean,yerr= y_greedy_div_std, fmt='.', label = "Greedy", color='green', linestyle = "-")
                      
                      
                      # axs3[i][j].set_xlabel("Num of sampled sequences")
                      # axs3[i][j].set_ylabel("Wall clock Time")
                      # axs3[i][j].plot(x,y_sample_time, marker='o',label="Ancestral sampling ", color='blue')
                      # axs3[i][j].plot(x,y_arit_time, marker='o', label = "Arithemetic sampling", color='green')
                      # axs3[i][j].plot(x,y_greedy_time, marker='o',label = "Greedy", color='red')
                      
                      c = c+1
                      
                      axs1[i][j].set_title(f"T={temp}, top-p={p}, top-k={k}, eps={eps}, \n p-values: Max = {np.round(p_test_max_bleu.pvalue,5)}, MBR = {np.round(p_test_mbr_bleu.pvalue,5)}", fontsize=8)
                      axs2[i][j].set_title(f"T={temp}, top-p={p}, top-k={k}, eps={eps}", fontsize=8)
                      # axs3[i][j].set_title(f"Temp {temp}, Top p {p}, Top k {k}, Eps {eps}")
                      
                      axs1[i][j].grid(True)
                      axs2[i][j].grid(True)
                      # axs3[i][j].grid(True)

                      # if c==16:
                      #     axs1[i][j].legend(loc =4, fontsize=6)
                      #     axs2[i][j].legend(loc =4, fontsize=8)
                          # axs3[i][j].legend(loc =4, fontsize=8)
                          
                    
            fig1.suptitle(f"BLEU: Model {model_name}, shot {shot}, Samples {samplesize}")
            fig2.suptitle(f"Div: Model {model_name}, shot {shot}, Samples {samplesize}")
            # fig3.suptitle(f"Time: Model {model_name}, shot {shot}, Samples {samplesize}")
            fig1.tight_layout()
            fig2.tight_layout()
            fig3.tight_layout()
            fig1.savefig(f"{output_dir}/Figure_BLEU_{model_name}__{shot}__Samples_{samplesize}.jpg")
            fig2.savefig(f"{output_dir}/Figure_Div_{model_name}__{shot}__Samples_{samplesize}.jpg") 
            fig1.savefig(f"{output_dir}/Figure_BLEU_{model_name}__{shot}__Samples_{samplesize}.svg")
            fig2.savefig(f"{output_dir}/Figure_Div_{model_name}__{shot}__Samples_{samplesize}.svg") 
            # fig3.savefig(f"{output_dir}/Figure_Time_{model_name}__{shot}__Samples_{samplesize}.jpg")                   
            plt.show()
            
            
    
    # it versions and quantized
    # for model_name in ["flan-t5"]:  #["llama-2", "flan-t5", "gemma", "mt5", "m2m", "mt0"]
    for model_name in ["flan-t5","mt0"]:
        for shot in ["zeroshot"]:  # ["zeroshot", "oneshot"]
            print(f"{model_name}, {shot}")
            # run_experiments(model_name, data, batch, output_dir, shot, load_in_8bit=False)
            
            
            # # tabular output resuls from json
            # json_data = []
            # for file in os.listdir(output_dir):
            #     if model_name in file and shot in file and "results" in file and str(samplesize)+"_" in file:
            #         with open(os.path.join(output_dir,file),"r") as f:
            #             jsondata = json.load(f)
            #             json_data.append(jsondata)
            # df = pd.DataFrame(json_data)
            # df.to_csv(f"{output_dir}/Table_{model_name}__{shot}__Samples_{samplesize}.csv")
            
            df = pd.read_csv(f"{output_dir}/Table_{model_name}__{shot}__Samples_{samplesize}.csv")
    
    
           
            
            for temp,p,k,eps in [[1,0.8,50,0.02]]: # temperature

            # for temp,p,k,eps in [[0.1,1.0,30,0.02],[0.1,1.0,50,0.02],[1,0.8,30,0.02],[1,1.0,50,0.05]]: # temperature
                     
                      # graphs
                      fig1, axs1 = plt.subplots(figsize=(10, 10))
                      fig2, axs2 = plt.subplots(figsize=(10, 10))
                    
                      # p test 
                      
                      with open(f'{output_dir}/{model_name}__{shot}__Samples_{samplesize}__N_40__temp_{temp}__p_{p}__k_{k}__eps{eps}_output.json') as f:
                          output_dict = json.load(f)
                          
                      ptest_max_bleu_arith_toppk = []
                      ptest_max_bleu_temp_toppk = []
                      ptest_mbr_bleu_arith_toppk = []
                      ptest_mbr_bleu_temp_toppk = []                    
                     
                      for idx in range(samplesize):   
                          idx = str(idx)
                          ptest_max_bleu_arith_toppk.append(float(output_dict[idx]['max_bleu_score_arith_toppk']))
                          ptest_max_bleu_temp_toppk.append(float(output_dict[idx]['max_bleu_score_temp_toppk']))
                         
                          ptest_mbr_bleu_arith_toppk.append(float(output_dict[idx]['mbr_bleu_score_arith_toppk']))
                          ptest_mbr_bleu_temp_toppk.append(float(output_dict[idx]['mbr_bleu_score_temp_toppk']))
                         
                      p_test_max_bleu = stats.ttest_rel(ptest_max_bleu_arith_toppk, ptest_max_bleu_temp_toppk)
                      p_test_mbr_bleu = stats.ttest_rel(ptest_mbr_bleu_arith_toppk, ptest_mbr_bleu_temp_toppk)
                     
                      # print(f"Model: {model_name}, p-test Max BLEU {p_test_max_bleu.pvalue}, p-test MBR BLEU {p_test_mbr_bleu.pvalue} \n")
                      
                  
                    
                  
                      rows = df[(df["temperature"]==temp) & (df["Top p "]==p) & (df["Top k"]==k) & (df["Epsilon"]==eps)].sort_values("Num decodes")
                      x = rows["Num decodes"]
                      y_max_sample_bleu_mean = rows["Max BLEU Sampling Mean"] 
                      y_max_sample_bleu_std = rows["Max BLEU Sampling Std"] 
                      y_max_arit_bleu_mean = rows["Max BLEU Arithmetic Mean"]
                      y_max_arit_bleu_std = rows["Max BLEU Arithmetic Std"]
                      y_max_greedy_bleu_mean = rows["Max BLEU Greedy Mean"]
                      y_max_greedy_bleu_std = rows["Max BLEU Greedy Std"]
                      
                      y_mbr_sample_bleu_mean = rows["MBR BLEU Sampling Mean"] 
                      y_mbr_sample_bleu_std = rows["MBR BLEU Sampling Std"] 
                      y_mbr_arit_bleu_mean = rows["MBR BLEU Arithmetic Mean"]
                      y_mbr_arit_bleu_std = rows["MBR BLEU Arithmetic Std"]
                      y_mbr_greedy_bleu_mean = rows["MBR BLEU Greedy Mean"]
                      y_mbr_greedy_bleu_std = rows["MBR BLEU Greedy Std"]
                      
                      y_sample_div_mean = rows["N gram diversity Sampling Mean"] 
                      y_sample_div_std = rows["N gram diversity Sampling Std"] 
                      y_arit_div_mean = rows["N gram diversity Arithmetic Mean"] 
                      y_arit_div_std = rows["N gram diversity Arithmetic Std"] 
                      y_greedy_div_mean = rows["N gram diversity Greedy Mean"]
                      y_greedy_div_std = rows["N gram diversity Greedy Std"]
                      
                      y_sample_time = rows["Time Sampling"] 
                      y_arit_time = rows["Time Arithmetic"] 
                      y_greedy_time = rows["Time Greedy"]
                      
                      # print(f"Arithmetic {np.round(y_mbr_arit_bleu_mean,2)}, {np.round(y_mbr_arit_bleu_std,2)}")
                      # print(f"Ancestral {np.round(y_mbr_sample_bleu_mean,2)},  {np.round(y_mbr_sample_bleu_std,2)}")
                      # CI
                      ci_sample =  y_mbr_sample_bleu_std
                      ci_arit = y_mbr_arit_bleu_std
                      
                      
                      axs1.set_xlabel("Number of sampled sequences", fontsize=30)
                      axs1.set_ylabel("BLEU score", fontsize=30)
    
                      # axs1.errorbar(x,y_max_sample_bleu_mean, yerr=y_max_sample_bleu_std, fmt='o',label="Ancestral sampling (Max) ", color='darkorange', linestyle = "--")
                      # axs1.errorbar(x,y_max_arit_bleu_mean, yerr=y_max_arit_bleu_std, fmt='^',label = "Arithemetic sampling (Max)", color='purple',  linestyle = "--")
                      
                      
                      axs1.plot(x,y_mbr_sample_bleu_mean, 'o', markersize=10,label="Ancestral sampling (MBR) ", color='darkorange', linestyle = "-")
                      axs1.fill_between(x, y_mbr_sample_bleu_mean-ci_sample, y_mbr_sample_bleu_mean+ci_sample, color='darkorange', alpha=0.15)
                      axs1.plot(x,y_mbr_arit_bleu_mean,'^', markersize=10, label = "Arithemetic sampling (MBR)", color='purple',  linestyle = "-")
                      axs1.fill_between(x, y_mbr_arit_bleu_mean-ci_arit, y_mbr_arit_bleu_mean+ci_arit, color='purple', alpha=0.15)
                      
                      axs1.errorbar(x,y_max_greedy_bleu_mean,yerr=y_max_greedy_bleu_std, label = "Greedy" ,color='green',  linestyle = "-")
                      
                      
                      axs2.set_xlabel("Number of sampled sequences", fontsize=30)
                      axs2.set_ylabel("N gram diversity", fontsize=30)
                      axs2.errorbar(x,y_sample_div_mean, yerr=y_sample_div_std,  fmt='o',markersize=10,label="Ancestral sampling ", color='darkorange', linestyle = "-")
                      axs2.errorbar(x,y_arit_div_mean, yerr= y_arit_div_std, fmt='^',markersize=10,label = "Arithemetic sampling", color='purple', linestyle = "-")
                      axs2.errorbar(x,y_greedy_div_mean,yerr= y_greedy_div_std, label = "Greedy", color='green', linestyle = "-")
                      
                      
                                      
                      axs1.set_title(f"T={temp}, top-p={p}, top-k={k}, eps={eps}, \n p-value MBR = {np.round(p_test_mbr_bleu.pvalue,5)}", fontsize=30)
                      axs2.set_title(f"T={temp}, top-p={p}, top-k={k}, eps={eps}", fontsize=30)
                      
                      axs1.grid(True)
                      axs2.grid(True)
                     
                    
                      # fig1.suptitle(f"BLEU: Model {model_name}, shot {shot}, Samples {samplesize}", fontsize=18)
                      # fig2.suptitle(f"Div: Model {model_name}, shot {shot}, Samples {samplesize}", fontsize=18)
                      fig1.tight_layout()
                      fig2.tight_layout()
                      fig1.savefig(f"outputs/WMT/V11/Fr-En_Figure_BLEU_{model_name}__{shot}__Samples_{samplesize}_temp_{temp}__p_{p}__k_{k}__eps{eps}.jpg")
                      fig2.savefig(f"outputs/WMT/V11/Fr-En_Figure_Div_{model_name}__{shot}__Samples_{samplesize}_temp_{temp}__p_{p}__k_{k}__eps{eps}.jpg") 
                      fig1.savefig(f"outputs/WMT/V11/Fr-En_Figure_BLEU_{model_name}__{shot}__Samples_{samplesize}_temp_{temp}__p_{p}__k_{k}__eps{eps}.svg")
                      fig2.savefig(f"outputs/WMT/V11/Fr-En_Figure_Div_{model_name}__{shot}__Samples_{samplesize}_temp_{temp}__p_{p}__k_{k}__eps{eps}.svg") 
                      plt.show()
                                     
    # c1 = 1.96 * y_max_sample_bleu_std/np.sqrt(20)
    # c2 = 1.96 * y_max_arit_bleu_std/np.sqrt(8)
    # plt.fill_between(x,(y_max_sample_bleu_mean-c1), (y_max_sample_bleu_mean+c1), label="Ancestral sampling (Max) ", color='darkblue', linestyle = "--")
    # plt.fill_between(x,(y_max_arit_bleu_mean-c2), (y_max_arit_bleu_mean+c2),label = "Arithemetic sampling (Max)", color='darkgreen',  linestyle = "-")
    
    
 