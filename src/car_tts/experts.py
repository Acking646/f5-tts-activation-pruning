"""Activation profiling and structural FFN pruning for F5-TTS DiT blocks.

The paper profiles token-level experts after clustering FFN neurons. Here an
expert is a balanced contiguous assignment in cosine/K-means space over the
first FFN projection. Profiling is post-GELU and selection is global after
per-layer Z-score normalization. Structural pruning reduces both FFN linear
matrices; merely applying a zero mask would not establish acceleration.
"""

import json
from pathlib import Path


def _torch():
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("Activation experiments require torch; install car-tts[tts]") from exc
    return torch


def find_feedforwards(model):
    torch = _torch()
    found = {}
    for name, module in model.named_modules():
        ff = getattr(module, "ff", None)
        if not isinstance(ff, torch.nn.Sequential) or len(ff) < 3:
            continue
        project = ff[0]
        if isinstance(project, torch.nn.Sequential) and project and isinstance(project[0], torch.nn.Linear):
            if isinstance(ff[-1], torch.nn.Linear):
                found[name] = module
    if not found:
        raise ValueError("No F5-TTS FeedForward modules found; check the upstream version")
    return found


def balanced_clusters(weight, n_experts=64, iterations=12, seed=42):
    """Balanced K-means labels for [neurons, input_dim] projection weights."""
    torch = _torch()
    count = weight.shape[0]
    if n_experts < 1 or n_experts > count or count % n_experts:
        raise ValueError("n_experts must evenly divide the FFN width")
    data = torch.nn.functional.normalize(weight.detach().float().cpu(), dim=1)
    generator = torch.Generator().manual_seed(seed)
    centers = data[torch.randperm(count, generator=generator)[:n_experts]].clone()
    capacity = count // n_experts
    labels = torch.empty(count, dtype=torch.long)
    for _ in range(iterations):
        similarities = data @ centers.T
        preferences = torch.argsort(similarities.flatten(), descending=True)
        labels.fill_(-1)
        sizes = torch.zeros(n_experts, dtype=torch.long)
        for flat in preferences.tolist():
            neuron, expert = divmod(flat, n_experts)
            if labels[neuron] < 0 and sizes[expert] < capacity:
                labels[neuron] = expert
                sizes[expert] += 1
                if bool((labels >= 0).all()):
                    break
        centers = torch.stack([data[labels == expert].mean(0) for expert in range(n_experts)])
        centers = torch.nn.functional.normalize(centers, dim=1)
    return labels


class ExpertProfiler:
    def __init__(self, model, n_experts=64, top_fraction=0.2, seed=42):
        self.modules = find_feedforwards(model)
        self.n_experts = n_experts
        self.top_fraction = top_fraction
        self.labels = {}
        self.pending = {}
        self.counts = {}
        self.total_positions = 0
        self.handles = []
        for offset, (name, module) in enumerate(self.modules.items()):
            labels = balanced_clusters(module.ff[0][0].weight, n_experts, seed=seed + offset)
            self.labels[name] = labels
            self.counts[name] = [0] * n_experts
            self.pending[name] = []
            self.handles.append(module.ff[0].register_forward_hook(self._hook(name)))

    def _hook(self, name):
        torch = _torch()
        labels = self.labels[name]

        def collect(_module, _inputs, output):
            flat = output.detach().float().reshape(-1, output.shape[-1]).cpu().abs()
            scores = torch.stack([flat[:, labels == expert].sum(1) for expert in range(self.n_experts)], dim=1)
            self.pending[name].append(scores)

        return collect

    def commit_sample(self):
        torch = _torch()
        calls = {len(value) for value in self.pending.values()}
        if not calls or len(calls) != 1:
            raise RuntimeError("FFN hook counts differ across layers: %s" % sorted(calls))
        for call_index in range(calls.pop()):
            normalized = []
            names = list(self.modules)
            for name in names:
                score = self.pending[name][call_index]
                normalized.append((score - score.mean(1, keepdim=True)) / (score.std(1, keepdim=True) + 1e-6))
            joined = torch.cat(normalized, dim=1)
            keep = max(1, int(joined.shape[1] * self.top_fraction))
            selected = torch.topk(joined, keep, dim=1).indices
            self.total_positions += joined.shape[0]
            for layer_index, name in enumerate(names):
                local = selected[(selected >= layer_index * self.n_experts) & (selected < (layer_index + 1) * self.n_experts)]
                local = local - layer_index * self.n_experts
                histogram = torch.bincount(local, minlength=self.n_experts).tolist()
                self.counts[name] = [left + right for left, right in zip(self.counts[name], histogram)]
        for name in self.pending:
            self.pending[name].clear()

    def close(self):
        for handle in self.handles:
            handle.remove()
        self.handles.clear()

    def result(self):
        return {
            "format": "car-tts-expert-profile-v1", "n_experts": self.n_experts,
            "top_fraction": self.top_fraction, "total_positions": self.total_positions,
            "layers": {name: {"labels": self.labels[name].tolist(), "activation_counts": self.counts[name]}
                       for name in self.modules},
        }


def make_masks(profile, retain_fraction=0.8, strategy="global"):
    layers = profile["layers"]
    n_experts = profile["n_experts"]
    masks = {name: [False] * n_experts for name in layers}
    if strategy == "equal":
        keep = max(1, round(n_experts * retain_fraction))
        for name, value in layers.items():
            ranked = sorted(range(n_experts), key=lambda index: value["activation_counts"][index], reverse=True)
            for index in ranked[:keep]:
                masks[name][index] = True
    elif strategy == "global":
        ranked = sorted(((count, name, index) for name, value in layers.items()
                         for index, count in enumerate(value["activation_counts"])), reverse=True)
        keep = max(len(layers), round(len(ranked) * retain_fraction))
        for _count, name, index in ranked[:keep]:
            masks[name][index] = True
        for name, value in layers.items():
            if not any(masks[name]):
                best = max(range(n_experts), key=lambda index: value["activation_counts"][index])
                masks[name][best] = True
    else:
        raise ValueError("strategy must be global or equal")
    return masks


def structurally_prune(model, profile, masks):
    torch = _torch()
    modules = find_feedforwards(model)
    before = sum(parameter.numel() for parameter in model.parameters())
    report = {}
    for name, expert_mask in masks.items():
        module = modules[name]
        first = module.ff[0][0]
        last = module.ff[-1]
        labels = torch.tensor(profile["layers"][name]["labels"], device=first.weight.device)
        chosen = torch.tensor(expert_mask, dtype=torch.bool, device=first.weight.device)[labels]
        indices = torch.nonzero(chosen, as_tuple=True)[0]
        new_first = torch.nn.Linear(first.in_features, len(indices), bias=first.bias is not None,
                                    device=first.weight.device, dtype=first.weight.dtype)
        new_last = torch.nn.Linear(len(indices), last.out_features, bias=last.bias is not None,
                                  device=last.weight.device, dtype=last.weight.dtype)
        with torch.no_grad():
            new_first.weight.copy_(first.weight[indices])
            if first.bias is not None:
                new_first.bias.copy_(first.bias[indices])
            new_last.weight.copy_(last.weight[:, indices])
            if last.bias is not None:
                new_last.bias.copy_(last.bias)
        module.ff[0][0] = new_first
        module.ff[-1] = new_last
        report[name] = {"original_neurons": first.out_features, "retained_neurons": len(indices)}
    after = sum(parameter.numel() for parameter in model.parameters())
    return {"parameters_before": before, "parameters_after": after,
            "parameter_reduction": 1 - after / before, "layers": report}


def load_profile(path):
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)

