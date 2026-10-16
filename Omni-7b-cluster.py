import os
# 设置Hugging Face镜像站点
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
# 增加缓存目录设置
os.environ['HF_HOME'] = '/root/autodl-tmp/huggingface_cache'
from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor
import torch
import argparse
import tqdm
import utils

parser = argparse.ArgumentParser()
parser.add_argument('--model_name', type=str, default='Qwen2.5-Omni-7b', help='model name in huggingface model hub')
parser.add_argument('--res_path', type=str, default='results/Qwen2.5-Omni-7b/', help='path to store the results of moefication')
parser.add_argument('--num-layer', type=int, default=28, help='number of layers')
parser.add_argument('--num-expert', type=int, default=296, help='number of experts')
parser.add_argument('--templates', type=str, default='thinker.model.layers.{}.mlp.up_proj.weight,talker.model.layers.{}.mlp.up_proj.weight', help='weight names of the first linear layer in each FFN (use comma to separate multiple templates)')

args = parser.parse_args()
if not os.path.exists(args.res_path):
    os.makedirs(args.res_path)

model_path = "/root/autodl-tmp/model/Qwen2.5-Omni-7B"
model = Qwen2_5OmniForConditionalGeneration.from_pretrained(model_path, torch_dtype=torch.float16, device_map="auto",)
processor = Qwen2_5OmniProcessor.from_pretrained(model_path)

torch.save(model.state_dict(), os.path.join(args.res_path, 'model.pt'))

config = utils.ModelConfig(os.path.join(args.res_path, 'model.pt'), args.res_path, split_num=args.num_expert)

template = args.templates.split(',')[0]

for i in tqdm.tqdm(range(23, args.num_layer)):
    print(f"当前处理层: {i}")  # 添加这行来确认起始值
    split = utils.ParamSplit(config, template, i)
    split.split()
    split.cnt()
    split.save()


