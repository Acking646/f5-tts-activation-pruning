import os
# 设置Hugging Face镜像站点
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
# 增加缓存目录设置
os.environ['HF_HOME'] = '/root/autodl-tmp/huggingface_cache'
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

model_path = "/root/autodl-tmp/model/llava-next-8b"
dataset_path = "/root/autodl-tmp/dataset/GQA"

data = load_dataset("nielsr/docvqa_1200_examples", split='test')

processor = LlavaNextProcessor.from_pretrained(model_path)
model = LlavaNextForConditionalGeneration.from_pretrained(model_path, torch_dtype=torch.float16, device_map="auto")


def change_forward_vision(model):

    def _forward(ffn_self, hidden_states):
        hidden_states = ffn_self.fc1(hidden_states)
        hidden_states = ffn_self.activation_fn(hidden_states)

        if ffn_self.patterns is not None:
            # golden
            bsz, seq_len, hidden_size = hidden_states.shape
            hidden_states_relu = hidden_states.clone()
            hidden_states_relu = hidden_states_relu.view(-1, hidden_size)
            score = torch.matmul(hidden_states_relu, ffn_self.patterns.transpose(0, 1))
            score = score.mean(dim=0)
            ffn_self.res.append(score.detach().cpu())
             
        hidden_states = ffn_self.fc2(hidden_states)
        return hidden_states

    def modify_ffn(ffn, path, res):
        assert type(ffn) == transformers.models.clip.modeling_clip.CLIPMLP
        labels = torch.load(path)
        cluster_num = max(labels)+1
        patterns = []
        for i in range(cluster_num):
            patterns.append(np.array(labels) == i)
        ffn.patterns = torch.Tensor(patterns).cuda().half()
        ffn.res = res
        ffn.forward_old = ffn.forward
        ffn.forward = types.MethodType(_forward, ffn)
    
    res_vision = {}
    # vision
    for layer_idx, layer in enumerate(model.vision_tower.vision_model.encoder.layers):
        ffn = layer.mlp
        path = os.path.join('results/llava-next-8b', 'param_split', 'vision_tower.vision_model.encoder.layers.{}.mlp.fc1.weight'.format(layer_idx))
        name = 'vision.layers.{}'.format(layer_idx)
        res_vision[name] = []
        modify_ffn(ffn, path, res_vision[name])

    return res_vision

def change_forward_langua(model):

    def _forward(ffn_self, hidden_states):

        gate_signal = ffn_self.gate_proj(hidden_states)
        gate_signal = ffn_self.act_fn(gate_signal)

        hidden_states = ffn_self.up_proj(hidden_states)
        hidden_states = hidden_states * gate_signal

        if ffn_self.patterns is not None:
            # golden
            bsz, seq_len, hidden_size = hidden_states.shape
            hidden_states_relu = hidden_states.clone()
            hidden_states_relu = hidden_states_relu.view(-1, hidden_size)
            score = torch.matmul(hidden_states_relu, ffn_self.patterns.transpose(0, 1))
            # Z-score标准化
            score = (score - score.mean(dim=1, keepdim=True)) / (score.std(dim=1, keepdim=True))
            ffn_self.res.append(score.detach().cpu())

        hidden_states = ffn_self.down_proj(hidden_states)

        return hidden_states

    def modify_ffn(ffn, path, res):
        assert type(ffn) == transformers.models.llama.modeling_llama.LlamaMLP
        labels = torch.load(path)
        cluster_num = max(labels)+1
        patterns = []
        for i in range(cluster_num):
            patterns.append(np.array(labels) == i)
        ffn.patterns = torch.Tensor(patterns).cuda().half()
        ffn.res = res
        ffn.forward_old = ffn.forward
        ffn.forward = types.MethodType(_forward, ffn)
    
    res_langua = {}
    # language
    for layer_idx, layer in enumerate(model.language_model.model.layers):
        ffn = layer.mlp
        path = os.path.join('results/llava-next-8b', 'param_split', 'language_model.model.layers.{}.mlp.up_proj.weight'.format(layer_idx))
        name = 'language.layers.{}'.format(layer_idx)
        res_langua[name] = []
        modify_ffn(ffn, path, res_langua[name])
    
    return res_langua

def topk_count(res, mat_count, token_count):
    res_stack = torch.stack([torch.cat(v, dim=0) for v in res.values()], dim=1)
    
    token, layer, expert = res_stack.shape
    token_count += token
    
    # 批量计算topk
    flat_activations = res_stack.view(-1, layer * expert)
    _, top_indices = torch.topk(flat_activations, k=int(layer * expert * 0.1), dim=1) # flat_activations.abs()
    
    # 使用bincount进行快速统计
    expert_indices = top_indices % expert
    layer_indices = top_indices // expert
    
    for l in range(layer):
        for e in range(expert):
            mat_count[l, e] += ((layer_indices == l) & (expert_indices == e)).sum().item()
    
    return mat_count, token_count


def heat_l(act_freq):
    # 转换为 numpy 数组
    data = np.array(act_freq)
    # 绘制热图
    plt.figure(figsize=(12, 2.4))
    plt.imshow(data, aspect='auto', cmap='Reds', vmin=data.min(), vmax=data.max())
    # 设置颜色条
    cbar = plt.colorbar(ticks=np.linspace(data.min(), data.max(), num=5))
    cbar.set_ticklabels([f'{tick:.2f}' for tick in np.linspace(data.min(), data.max(), num=5)])
    cbar.ax.tick_params(labelsize=7)  # 设置颜色条刻度字体大小
    # 移除颜色条边界框
    cbar.outline.set_visible(False)
    # 设置坐标轴
    plt.xticks(range(0, data.shape[1], 10), range(0, data.shape[1], 10))
    plt.tick_params(axis='x', labelsize=8)  # 设置横坐标标签字体大小
    plt.yticks(range(0, data.shape[0], 2), range(0, data.shape[0], 2))  # 添加这行设置纵坐标刻度
    plt.tick_params(axis='y', labelsize=6.7)  # 设置纵坐标标签字体大小
    # 移除所有刻度线
    plt.tick_params(axis='both', which='both', length=0)  # 设置刻度线长度为0
    # 设置边界框为无色
    for spine in plt.gca().spines.values():
        spine.set_visible(False)
    plt.xlabel('Experts')
    plt.ylabel('Layers')
    plt.title('LLaVa-NeXT-8B Expert activation frequency for Vision')
    # 保存并显示图像
    plt.tight_layout()
    plt.savefig(f'figure/llava-8b_image.png')
    plt.close()
    del data


def heat_v(act):
    # 转换为 numpy 数组
    data = np.array(act)
    # 绘制热图
    plt.figure(figsize=(8, 2))
    plt.imshow(data, aspect='auto', cmap='Reds', vmin=data.min(), vmax=data.max())
    # 设置颜色条
    cbar = plt.colorbar(ticks=np.linspace(data.min(), data.max(), num=5))
    cbar.set_ticklabels([f'{tick:.2f}' for tick in np.linspace(data.min(), data.max(), num=5)])
    cbar.ax.tick_params(labelsize=7)  # 设置颜色条刻度字体大小
    # 移除颜色条边界框
    cbar.outline.set_visible(False)
    # 设置坐标轴
    plt.xticks(range(0, data.shape[1], 10), range(0, data.shape[1], 10))
    plt.tick_params(axis='x', labelsize=8)  # 设置横坐标标签字体大小
    plt.yticks(range(0, data.shape[0], 2), range(0, data.shape[0], 2))  # 添加这行设置纵坐标刻度
    plt.tick_params(axis='y', labelsize=6.7)  # 设置纵坐标标签字体大小
    # 移除所有刻度线
    plt.tick_params(axis='both', which='both', length=0)  # 设置刻度线长度为0
    # 设置边界框为无色
    for spine in plt.gca().spines.values():
        spine.set_visible(False)
    plt.xlabel('Experts')
    plt.ylabel('Layers')
    plt.title('LLaVa-NeXT-8B Expert activation value for Vision Encoder')
    # 保存并显示图像
    plt.tight_layout()
    plt.savefig(f'figure/llava-8b_encoder_50.png')
    plt.close()
    del data


# 计数初始化
mat_count_l = torch.zeros(32, 256)
token_count_l = 0
# mat_count_v = torch.zeros(24, 128)
# token_count_v = 0

# 修改FFN层
res_langua = change_forward_langua(model)
res_vision = change_forward_vision(model)

# 遍历数据集中的每个样本
sample_count = 0  # 添加计数器
for sample in tqdm(data, leave=False):
    if sample_count >= 50:  # 处理200个样本后停止
        break
    # 清空之前的结果
    for v in res_langua.values():
        v.clear()

    # 准备图像和问题
    image = sample["image"]
    # question = sample["Translated Question"] # Translated Question
    # options = sample["Translated Options"] # Translated Options
    # full_question = f"{question}{options}"

    # 构建对话格式
    conversation = [
        {
        "role": "user",
        "content": [
            # {"type": "text", "text": full_question},
            {"type": "image"},
            ],
        },
    ]
    
    prompt = processor.apply_chat_template(conversation, add_generation_prompt=True) # 处理对话模板
    inputs = processor(images=image, text=prompt, return_tensors="pt").to(model.device) # 预处理图像和文本输入
    
    # 只进行前向传播，不生成输出
    with torch.no_grad():
        model(**inputs)

    # # 生成回答
    # output = model.generate(
    #     **inputs,
    #     max_new_tokens=1,
    #     pad_token_id=processor.tokenizer.eos_token_id # 使用结束标记作为填充标记
    # ) 

    # answer = processor.decode(output[0], skip_special_tokens=True)
    # print(answer)
    mat_count_l, token_count_l = topk_count(res_langua, mat_count_l, token_count_l)
    sample_count += 1  # 处理完一个样本后增加计数


avg = []
for k, v in res_vision.items():
    avg.append(torch.stack(v).mean(dim=0))
avg = torch.stack(avg)
heat_v(avg)

# 计算激活频率
act_freq_l = mat_count_l / token_count_l
torch.save(act_freq_l, 'figure/llava-8b_image.pt')
heat_l(act_freq_l)
