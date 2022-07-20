# -*- coding: utf-8 -*-
# file: Attention_Visualization.py
# time: 27/06/2022
# author: yangheng <yangheng@m.scnu.edu.cn>
# github: https://github.com/yangheng95
# Copyright (C) 2021. All Rights Reserved.

from transformers import AutoTokenizer, AutoModel, utils
from bertviz import model_view
utils.logging.set_verbosity_error()  # Suppress standard warnings

# model_name = "textattack/bert-base-uncased-sst2"
model_name = "howey/bert-base-uncased-sst2"
# Find popular HuggingFace models here: https://huggingface.co/models
input_text1 = "GM pulls Guy Ritchie driving ad after protest Protests from seven safety ists have spurred General Motors to pull a television ad that is a young ster driving a Corvette sports car so recklessly that it goes airborne, opponents of the automaker say."
input_text2 = "GM pulls Guy Ritchie driving ad after protest Protests from seven safety [MASK] have [MASK] General Motors to pull a television ad that [MASK] a young [MASK] driving a Corvette sports car so recklessly that it goes a[MASK] [MASK] of the a[MASK] say."
model = AutoModel.from_pretrained(model_name, output_attentions=True)  # Configure model to return attention values
tokenizer = AutoTokenizer.from_pretrained(model_name)
inputs = tokenizer.encode(input_text1, return_tensors='pt')  # Tokenize input text
outputs = model(inputs)  # Run model
attention = outputs[-1]  # Retrieve attention from model outputs
tokens = tokenizer.convert_ids_to_tokens(inputs[0])  # Convert input ids to token strings
model_view(attention, tokens)  # Display model view
