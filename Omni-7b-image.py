import os
# 设置Hugging Face镜像站点
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
# 增加缓存目录设置
os.environ['HF_HOME'] = '/root/autodl-tmp/huggingface_cache'
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


model_path = "/root/autodl-tmp/model/Qwen2.5-Omni-7B"
model = Qwen2_5OmniForConditionalGeneration.from_pretrained(model_path, torch_dtype=torch.float16, device_map="auto",)
processor = Qwen2_5OmniProcessor.from_pretrained(model_path)
model.disable_talker() # 禁用音频输出
data = load_dataset("lmms-lab/GQA", "challenge_all_images", split="challenge")

def change_forward_thinker(model):

    def _forward(ffn_self, hidden_states):

        gate_signal = ffn_self.gate_proj(hidden_states)
        gate_signal = ffn_self.act_fn(gate_signal)

        hidden_states = ffn_self.up_proj(hidden_states)
        hidden_states = hidden_states * gate_signal

        if ffn_self.patterns is not None:
            # golden
            bsz, seq_len, hidden_size = hidden_states.shape 
            if seq_len > 1:
                hidden_states_relu = hidden_states.clone()
                hidden_states_relu = hidden_states_relu.view(-1, hidden_size)
                score = torch.matmul(hidden_states_relu, ffn_self.patterns.transpose(0, 1)) 
                # Z-score标准化
                score = (score - score.mean(dim=1, keepdim=True)) / (score.std(dim=1, keepdim=True))
                ffn_self.res.append(score.detach().cpu())

        hidden_states = ffn_self.down_proj(hidden_states)

        return hidden_states

    def modify_ffn(ffn, path, res):
        assert type(ffn) == transformers.models.qwen2_5_omni.modeling_qwen2_5_omni.Qwen2MLP
        labels = torch.load(path, weights_only=False)
        cluster_num = max(labels)+1
        patterns = []
        for i in range(cluster_num):
            patterns.append(np.array(labels) == i)
        ffn.patterns = torch.Tensor(patterns).cuda().half()
        ffn.res = res
        ffn.forward_old = ffn.forward
        ffn.forward = types.MethodType(_forward, ffn)
    
    res_thinker = {}
    # language
    for layer_idx, layer in enumerate(model.thinker.model.layers):
        ffn = layer.mlp
        path = os.path.join('results/Qwen2.5-Omni-7b', 'param_split', 'thinker.model.layers.{}.mlp.up_proj.weight'.format(layer_idx))
        name = 'language.layers.{}'.format(layer_idx)
        res_thinker[name] = []
        modify_ffn(ffn, path, res_thinker[name])
    
    return res_thinker

def change_forward_talker(model):

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
        assert type(ffn) == transformers.models.qwen2_5_omni.modeling_qwen2_5_omni.Qwen2MLP
        labels = torch.load(path, weights_only=False) # 4096个标签
        cluster_num = max(labels)+1
        patterns = []
        for i in range(cluster_num):
            patterns.append(np.array(labels) == i)
        ffn.patterns = torch.Tensor(patterns).cuda().half() #[128, 4096]
        ffn.res = res
        ffn.forward_old = ffn.forward
        ffn.forward = types.MethodType(_forward, ffn)
    
    res_talker = {}
    # vision
    for layer_idx, layer in enumerate(model.talker.model.layers):
        ffn = layer.mlp
        path = os.path.join('results/Qwen2.5-Omni-7b', 'param_split', 'talker.model.layers.{}.mlp.up_proj.weight'.format(layer_idx))
        name = 'vision.layers.{}'.format(layer_idx)
        res_talker[name] = []
        modify_ffn(ffn, path, res_talker[name])

    return res_talker


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


def heat_thinker(act_freq):
    # 转换为 numpy 数组
    data = np.array(act_freq)
    # 绘制热图
    plt.figure(figsize=(12, 2))
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
    plt.title('Qwen2.5-Omni-7B Thinker expert activation frequency for Visual')
    # 保存并显示图像
    plt.tight_layout()
    plt.savefig(f'figure/Omni-7b_thinker_image.png')
    plt.close()
    del data

def heat_talker(act_freq):
    # 转换为 numpy 数组
    data = np.array(act_freq)
    # 绘制热图
    plt.figure(figsize=(12, 2))
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
    plt.title('Qwen2.5-Omni-7B Talker expert activation frequency for Visual')
    # 保存并显示图像
    plt.tight_layout()
    plt.savefig(f'figure/Omni-7b_talker_image.png')
    plt.close()
    del data


# 计数初始化
mat_count_thinker = torch.zeros(28, 296)
token_count_thinker = 0
# mat_count_talker = torch.zeros(24, 296)
# token_count_talker = 0

# 修改FFN层
res_thinker = change_forward_thinker(model)
# res_takler = change_forward_talker(model)

# 遍历数据集中的每个样本
sample_count = 0  # 添加计数器
for sample in tqdm(data, leave=False):
    if sample_count >= 1000:
        break
    # 清空之前的结果
    for v in res_thinker.values():
        v.clear()
    # for v in res_takler.values():
    #     v.clear()
    
    image = sample['image']

    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
            ],
        },
    ]

    # 是否提取视频中的音频
    USE_AUDIO_IN_VIDEO = False

    # Preparation for inference
    text = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
    audios, images, videos = process_mm_info(conversation, use_audio_in_video=USE_AUDIO_IN_VIDEO)
    inputs = processor(text=text, audio=audios, images=images, videos=videos, return_tensors="pt", padding=True, use_audio_in_video=USE_AUDIO_IN_VIDEO)
    inputs = inputs.to(model.device).to(model.dtype)
 
    # # Inference: Generation of the output text and audio
    text_ids = model.generate(**inputs, use_audio_in_video=USE_AUDIO_IN_VIDEO, return_audio=False, max_new_tokens=5)
    # text = processor.batch_decode(text_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    # print(text)

    mat_count_thinker, token_count_thinker = topk_count(res_thinker, mat_count_thinker, token_count_thinker)
    # mat_count_talker, token_count_talker = topk_count(res_takler, mat_count_talker, token_count_talker)
    sample_count += 1  # 处理完一个样本后增加计数

# 计算激活频率
act_thinker = mat_count_thinker / token_count_thinker
torch.save(act_thinker, f'figure/Omni-7b_thinker_image.pt')
heat_thinker(act_thinker)
# act_freq_takler = mat_count_talker / token_count_talker
# torch.save(act_freq_takler, f'figure/Omni-7b_talker_audio.pt')
# heat_talker(act_freq_takler)