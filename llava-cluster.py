import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration
import torch
import sys
import numpy as np
import argparse
import tqdm

sys.path.append('moefication')
import utils
parser = argparse.ArgumentParser()
parser.add_argument('--model_name', type=str, default='llava-next-8b', help='model name in huggingface model hub')
parser.add_argument('--res_path', type=str, default='results/llava-next-8b/', help='path to store the results of moefication')
parser.add_argument('--num-layer', type=int, default=24, help='number of layers')
parser.add_argument('--num-expert', type=int, default=128, help='number of experts')
parser.add_argument('--templates', type=str, default='language_model.model.layers.{}.mlp.up_proj.weight,vision_tower.vision_model.encoder.layers.{}.mlp.fc1.weight', help='weight names of the first linear layer in each FFN (use comma to separate multiple templates)')

args = parser.parse_args()
if not os.path.exists(args.res_path):
    os.makedirs(args.res_path)

processor = LlavaNextProcessor.from_pretrained("llava-hf/llama3-llava-next-8b-hf")
model = LlavaNextForConditionalGeneration.from_pretrained("llava-hf/llama3-llava-next-8b-hf", torch_dtype=torch.float16, device_map="auto") 

torch.save(model.state_dict(), os.path.join(args.res_path, 'model.pt'))

config = utils.ModelConfig(os.path.join(args.res_path, 'model.pt'), args.res_path, split_num=args.num_expert)

template = args.templates.split(',')[1]

for i in tqdm.tqdm(range(args.num_layer)):
    split = utils.ParamSplit(config, template, i)
    split.split()
    split.cnt()
    split.save()


