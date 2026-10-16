import os
# 设置Hugging Face镜像站点
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
# 增加缓存目录设置
os.environ['HF_HOME'] = '/root/autodl-tmp/huggingface_cache'
# 优化内存分配策略
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True,garbage_collection_threshold:0.8'
import types
import torch
from datasets import load_dataset
import numpy as np
import soundfile as sf # 处理音频文件
from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor
from qwen_omni_utils import process_mm_info # 处理多模态信息
import transformers
from PIL import Image
import requests
import json
import matplotlib.pyplot as plt
from tqdm import tqdm
import gc


model_path = "/root/autodl-tmp/model/Qwen2.5-Omni-7B"
model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
    model_path, 
    torch_dtype=torch.float16, 
    device_map="auto" # 改为平衡分配模式
    # max_memory={i: '46GiB' for i in range(torch.cuda.device_count())}  # 显式指定各GPU内存
)
processor = Qwen2_5OmniProcessor.from_pretrained(model_path)
# model.eval()  # 设置模型为评估模式
model.disable_talker() # 禁用音频输出


def change_forward_thinker(model, indices):

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
        assert type(ffn) == transformers.models.qwen2_5_omni.modeling_qwen2_5_omni.Qwen2MLP
        labels = torch.load(path, weights_only=False)
        cluster_num = max(labels)+1
        patterns = []
        for i in range(cluster_num):
            patterns.append(np.array(labels) == i)
        ffn.patterns = torch.Tensor(patterns).cuda().half()
        ffn.indice = indice
        ffn.forward_old = ffn.forward
        ffn.forward = types.MethodType(_forward, ffn)
    

    # language
    for layer_idx, layer in enumerate(model.thinker.model.layers):
        ffn = layer.mlp
        path = os.path.join('results/Qwen2.5-Omni-7b', 'param_split', 'thinker.model.layers.{}.mlp.up_proj.weight'.format(layer_idx))
        modify_ffn(ffn, path, indices[layer_idx])


# 加载两个张量
image = torch.load(f'figure/Omni-7b_thinker_image.pt')
text = torch.load(f'figure/Omni-7b_thinker_text.pt')
audio = torch.load(f'figure/Omni-7b_thinker_audio.pt')

# 计算模态共享专家
shared_mask = (
    (image > 0.05) &
    (text > 0.05) &
    (audio > 0.05) &
    (torch.abs(image - text) < 0.15) &
    (torch.abs(image - audio) < 0.15) &
    (torch.abs(text - audio) < 0.15)
)
modal_shared = torch.where(shared_mask, (image + text + audio) / 3, torch.zeros_like(image))
high_freq_mask = shared_mask & (modal_shared >= 0.25)
low_freq_mask = shared_mask & (modal_shared < 0.25)
high_freq_count = high_freq_mask.sum().item()
low_freq_count = low_freq_mask.sum().item()
print(f"高频共享专家: {high_freq_count}")
print(f"低频共享专家: {low_freq_count}")
print(f"模态共享专家: {high_freq_count + low_freq_count}")

# 计算模态特异性专家
freq = 0.15
# 创建专家分类矩阵
expert_types = torch.zeros_like(image)
image_expert = ((image - text) > freq) & ((image - audio) > freq)
text_expert = ((text - image) > freq) & ((text - audio) > freq)
audio_expert = ((audio - image) > freq) & ((audio - text) > freq)
# 计算各类专家数量
image_count = image_expert.sum().item()
text_count = text_expert.sum().item()
audio_count = audio_expert.sum().item()
print(f"image专家: {image_count}")
print(f"text专家: {text_count}") 
print(f"audio专家: {audio_count}")

expert = shared_mask | image_expert | text_expert
expert_num = expert.sum().item()
prun_rate = expert_num / (28 * 296)
print(f"expert_num: {expert_num}  prun_rate: {prun_rate}")
indices = [torch.nonzero(expert[i], as_tuple=True)[0] for i in range(expert.size(0))]


data = load_dataset("Lin-Chen/MMStar", split='val' )
change_forward_thinker(model, indices)

count = 0 
correct = 0
max = 0

# 遍历数据集中的每个样本
for sample in tqdm(data, leave=False):
    if count >= 1000: 
        break

    # # 在每次迭代前强制清理缓存
    # torch.cuda.empty_cache()
    # gc.collect()

    image = sample['image']
    question = f"{sample['question']}. Output option label 'A' 'B' 'C' 'D' only, Don't output other responses."

    conversation = [
            {
            "role": "system",
            "content": [
                {"type": "text", "text": "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable of perceiving auditory and visual inputs, as well as generating text and speech."}
            ],
        },
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": question},
            ],
        },
    ]

    # 是否提取视频中的音频
    USE_AUDIO_IN_VIDEO = False

    # Preparation for inference
    text = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
    _, images, _ = process_mm_info(conversation, use_audio_in_video=USE_AUDIO_IN_VIDEO) # audios, images, videos
    inputs = processor(text=text, images=images, return_tensors="pt", padding=True, use_audio_in_video=USE_AUDIO_IN_VIDEO)
    inputs = inputs.to(model.device).to(model.dtype)
 
    # # Inference: Generation of the output text and audio
    text_ids = model.generate(**inputs, use_audio_in_video=USE_AUDIO_IN_VIDEO, return_audio=False, max_new_tokens=5)
    text = processor.batch_decode(text_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)

    text = text[0].split("assistant")[-1].strip().split(".")[0]
    answer = sample["answer"] # 真实标签
    if text == answer:
        correct += 1
    count += 1
    acc = correct / count
    if acc > max and count > 20:
        max = acc
    print(f"max: {max}  acc: {acc}  output: {text}  answer: {answer}")

    # # 清理缓存
    del inputs, text_ids, text
    gc.collect()
    torch.cuda.empty_cache()