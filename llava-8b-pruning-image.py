import os
# 设置Hugging Face镜像站点
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
# 增加缓存目录设置
os.environ['HF_HOME'] = '/root/autodl-tmp/huggingface_cache'
# 优化内存分配策略
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True,garbage_collection_threshold:0.8'
import types
import torch
from datasets import load_dataset, load_from_disk
import numpy as np
from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration
import transformers
from PIL import Image
import requests
import json
import matplotlib.pyplot as plt
from collections import Counter
from tqdm import tqdm
import gc

model_path = "/root/autodl-tmp/model/llava-next-8b"
processor = LlavaNextProcessor.from_pretrained(model_path)
model = LlavaNextForConditionalGeneration.from_pretrained(model_path, torch_dtype=torch.float16, device_map="auto")


def change_forward_langua(model, indices):

    def _forward(ffn_self, hidden_states):

        gate_signal = ffn_self.gate_proj(hidden_states)
        gate_signal = ffn_self.act_fn(gate_signal)

        hidden_states = ffn_self.up_proj(hidden_states)
        hidden_states = hidden_states * gate_signal

        if ffn_self.patterns is not None:
            # golden
            bsz, seq_len, hidden_size = hidden_states.shape
            if seq_len > 1:
                labels = ffn_self.indice
                labels = labels.to(ffn_self.patterns.device)
                labels = labels.unsqueeze(0).unsqueeze(0)
                labels = labels.expand(bsz, seq_len, -1)
                cur_mask = torch.nn.functional.embedding(labels, ffn_self.patterns).sum(-2)
                hidden_states[cur_mask == False] = 0  

        hidden_states = ffn_self.down_proj(hidden_states)

        return hidden_states

    def modify_ffn(ffn, path, indice):
        assert type(ffn) == transformers.models.llama.modeling_llama.LlamaMLP
        labels = torch.load(path) # 14336个标签
        cluster_num = max(labels)+1
        patterns = []
        for i in range(cluster_num):
            patterns.append(np.array(labels) == i)
        ffn.patterns = torch.Tensor(patterns).cuda().half()
        ffn.indice = indice
        ffn.forward_old = ffn.forward
        ffn.forward = types.MethodType(_forward, ffn)
    
    # language
    for layer_idx, layer in enumerate(model.language_model.model.layers):
        ffn = layer.mlp
        path = os.path.join('results/llava-next-8b', 'param_split', 'language_model.model.layers.{}.mlp.up_proj.weight'.format(layer_idx))
        modify_ffn(ffn, path, indices[layer_idx])
    

# 加载两个张量
image = torch.load(f'figure/llava-8b_image.pt')
text = torch.load(f'figure/llava-8b_text.pt')

# 计算模态共享专家
shared_mask = ((image > 0.05) &
               (text > 0.05) & 
               (torch.abs(image - text) < 0.15)
)
modal_shared = torch.where(shared_mask, (image + text) / 2, torch.zeros_like(image))
high_shared_mask = shared_mask & (modal_shared >= 0.25)
low_shared_mask = shared_mask & (modal_shared < 0.25)
high_freq_count = high_shared_mask.sum().item()
low_freq_count = low_shared_mask.sum().item()
print(f"高频共享专家: {high_freq_count}")
print(f"低频共享专家: {low_freq_count}")
print(f"模态共享专家: {high_freq_count + low_freq_count}")

freq = 0.15
image_expert = (image - text) > freq
text_expert = (text - image) > freq

image_count = image_expert.sum().item()
text_count = text_expert.sum().item()
print(f"image专家数量: {image_count}")
print(f"text专家数量: {text_count}") 

expert = shared_mask | image_expert

expert_num = expert.sum().item()
prun_rate = expert_num / (32 * 256)
print(f"expert_num: {expert_num}  prun_rate: {prun_rate}")
indices = [torch.nonzero(expert[i], as_tuple=True)[0] for i in range(expert.size(0))]


data = load_dataset("Lin-Chen/MMStar", split='val' )
change_forward_langua(model, indices)

count = 0 
correct = 0
max = 0
# 遍历数据集中的每个样本
for sample in tqdm(data, leave=False):
    if count >= 1000: 
        break

    image = sample['image']
    question = f"{sample['question']}. Output option label 'A' 'B' 'C' 'D' only, Don't output other responses."

    # 构建对话格式
    conversation = [
        {
        "role": "user",
        "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": question},
            
            ],
        },
    ]
    
    prompt = processor.apply_chat_template(conversation, add_generation_prompt=True) # 处理对话模板
    inputs = processor(images=image, text=prompt, return_tensors="pt").to(model.device) # 预处理图像和文本输入
    
    # 生成回答
    output = model.generate(
        **inputs,
        max_new_tokens=2,
        pad_token_id=processor.tokenizer.eos_token_id # 使用结束标记作为填充标记
    ) 

    text = processor.decode(output[0], skip_special_tokens=True)

    text = text.split("assistant")[-1].strip()
    answer = sample["answer"] # 真实标签
    if text == answer:
        correct += 1
    count += 1
    acc = correct / count
    if acc > max and count > 20:
        max = acc
    print(f"max: {max}  acc: {acc}  output: {text}  answer: {answer}")
    # # 清理缓存
    gc.collect()
    torch.cuda.empty_cache()
