from dataclasses import dataclass
import csv
import json
import os
from typing import Generator, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.cluster import SpectralClustering
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer



    

@dataclass
class ModelLayerConfig:
    model_name: str
    model_path: str
    layer_idx: int


class ModelBundle:
    def __init__(self, cfg: ModelLayerConfig):
        self.model_config = AutoConfig.from_pretrained(
            cfg.model_path,
            torch_dtype=torch.float16,
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            cfg.model_path,
            config=self.model_config,
            torch_dtype=torch.float16,
            device_map="auto",
        )
        self.model.eval()
        self.tokenizer = AutoTokenizer.from_pretrained(cfg.model_path)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token


@dataclass
class OfflineRes:
    model_layer_config: ModelLayerConfig
    tc_cc_ratio: float
    HCNS: np.ndarray
    LCNS: np.ndarray

    @staticmethod
    def load(save_root: str, model_layer_config: ModelLayerConfig) -> "OfflineRes":
        save_dir = os.path.join(
            save_root,
            model_layer_config.model_name,
            f"layer_{model_layer_config.layer_idx}",
        )
        with open(os.path.join(save_dir, "meta.json"), "r", encoding="utf-8") as f:
            meta = json.load(f)
        return OfflineRes(
            model_layer_config=model_layer_config,
            tc_cc_ratio=float(meta["tc_cc_ratio"]),
            HCNS=np.load(os.path.join(save_dir, "HCNS.npy")),
            LCNS=np.load(os.path.join(save_dir, "LCNS.npy")),
        )

    def save(self, save_root: str) -> None:
        save_dir = os.path.join(
            save_root,
            self.model_layer_config.model_name,
            f"layer_{self.model_layer_config.layer_idx}",
        )
        os.makedirs(save_dir, exist_ok=True)
        with open(os.path.join(save_dir, "meta.json"), "w", encoding="utf-8") as f:
            json.dump(
                {
                    "model_name": self.model_layer_config.model_name,
                    "layer_idx": self.model_layer_config.layer_idx,
                    "tc_cc_ratio": float(self.tc_cc_ratio),
                },
                f,
                indent=2,
                ensure_ascii=False,
            )
        np.save(os.path.join(save_dir, "HCNS.npy"), self.HCNS)
        np.save(os.path.join(save_dir, "LCNS.npy"), self.LCNS)
        print(f"[OfflineRes] Saved to {save_dir}")


class neurons2GPUSorver:
    @staticmethod
    def split_HCNS_LCNS_by_spectral(cw: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        # cw: (k, k) CPU numpy
        cw = cw.astype(np.float64)
        cw = 0.5 * (cw + cw.T)
        np.fill_diagonal(cw, np.maximum(np.diag(cw), 1e-6))

        sc = SpectralClustering(
            n_clusters=2,
            affinity="precomputed",
            assign_labels="kmeans",
            random_state=42,
            n_init=10,
        )
        labels = sc.fit_predict(cw)

        coact_scores = cw.sum(axis=1)
        mean0 = coact_scores[labels == 0].mean()
        mean1 = coact_scores[labels == 1].mean()

        if mean0 > mean1:
            hcns = np.where(labels == 0)[0]
            lcns = np.where(labels == 1)[0]
        else:
            hcns = np.where(labels == 1)[0]
            lcns = np.where(labels == 0)[0]
        return hcns, lcns


class neurons2TC_CCSorver:
    @staticmethod
    def find_ratio_by_history_idx(freqs: np.ndarray) -> float:
        split_HCNS_LCNS_by_history
        不再用这种方法，而是对于给定的x_list的每个X。遍历整个n_token维度，
        return float(np.mean(freqs))


class OfflineSolver:
    def __init__(self, model_layer_config: ModelLayerConfig, model_bundle: ModelBundle):
        self.model_layer_config = model_layer_config
        self.model_config = model_bundle.model_config
        self.model = model_bundle.model
        self.tokenizer = model_bundle.tokenizer

        self.fc1_module = self._resolve_fc1_module()
        self.fc1_device = self.fc1_module.weight.device
        self.input_device = self.model.get_input_embeddings().weight.device

    def _resolve_fc1_module(self):
        layer_idx = self.model_layer_config.layer_idx

        if hasattr(self.model, "transformer") and hasattr(self.model.transformer, "h"):
            return self.model.transformer.h[layer_idx].mlp.c_fc  # GPT2-like

        if hasattr(self.model, "model") and hasattr(self.model.model, "decoder"):
            return self.model.model.decoder.layers[layer_idx].fc1  # OPT-like

        raise KeyError(f"Cannot locate fc1 module for layer {layer_idx}")

    @staticmethod
    def _batch_iter(items: List[str], batch_size: int) -> Generator[List[str], None, None]:
        for i in range(0, len(items), batch_size):
            yield items[i : i + batch_size]

    def _read_prompts_from_csv(
        self,
        dataset_csv_path: str,
        max_samples: Optional[int],
        use_system_prompt: bool,
    ) -> List[str]:
        prompts: List[str] = []
        with open(dataset_csv_path, "r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            if not reader.fieldnames or "question" not in reader.fieldnames:
                raise ValueError(
                    f"{dataset_csv_path} 缺少 'question' 列。"
                    "请使用 small_openorca.csv，或改字段映射。"
                )

            for row in reader:
                question = (row.get("question") or "").strip()
                if not question:
                    continue
                if use_system_prompt:
                    system_prompt = (row.get("system_prompt") or "").strip()
                    prompt = f"{system_prompt}\n\n{question}" if system_prompt else question
                else:
                    prompt = question
                prompts.append(prompt)
                if max_samples is not None and len(prompts) >= max_samples:
                    break

        if not prompts:
            raise ValueError("CSV 中没有有效 question 样本。")
        return prompts

    def _capture_fc1_input_flat(
        self,
        prompts: List[str],
        n_token: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        返回:
        hidden_2d: [N, H] on fc1_device
        valid_mask_1d: [N] bool on fc1_device
        """
        inputs = self.tokenizer(
            prompts,
            padding="max_length",
            truncation=True,
            max_length=n_token,
            return_tensors="pt",
        )
        attention_mask = inputs["attention_mask"].to(self.fc1_device, non_blocking=True)

        # 输入放到 embedding 所在设备，交给 HF/accelerate 分发
        model_inputs = {
            "input_ids": inputs["input_ids"].to(self.input_device, non_blocking=True),
            "attention_mask": inputs["attention_mask"].to(self.input_device, non_blocking=True),
        }

        captured = {"x": None}

        def _pre_hook(_module, module_inputs):
            captured["x"] = module_inputs[0].detach()

        handle = self.fc1_module.register_forward_pre_hook(_pre_hook)
        try:
            with torch.inference_mode():
                _ = self.model(**model_inputs, return_dict=True, use_cache=False)
        finally:
            handle.remove()

        if captured["x"] is None:
            raise RuntimeError("Failed to capture FC1 input from forward pre-hook.")

        x = captured["x"]  # 可能是 [B,S,H]，也可能是 [B*S,H]
        bsz, seqlen = attention_mask.shape
        expected_n = bsz * seqlen

        if x.ndim == 3:
            hidden_2d = x.reshape(expected_n, x.shape[-1]).to(self.fc1_device, non_blocking=True)
            valid_mask_1d = attention_mask.reshape(expected_n).bool()
            return hidden_2d, valid_mask_1d

        if x.ndim == 2:
            # 修复你遇到的 bug：某些实现在 fc1 前就把 [B,S,H] 展平为 [B*S,H]
            hidden_2d = x.to(self.fc1_device, non_blocking=True)
            if hidden_2d.shape[0] == expected_n:
                valid_mask_1d = attention_mask.reshape(expected_n).bool()
                return hidden_2d, valid_mask_1d

            valid_tokens = int(attention_mask.sum().item())
            if hidden_2d.shape[0] == valid_tokens:
                valid_mask_1d = torch.ones(valid_tokens, dtype=torch.bool, device=self.fc1_device)
                return hidden_2d, valid_mask_1d

            raise RuntimeError(
                f"Hook tensor shape mismatch: x.shape={tuple(hidden_2d.shape)}, "
                f"expected B*S={expected_n}, valid_tokens={valid_tokens}"
            )

        raise RuntimeError(f"Unexpected hook tensor ndim={x.ndim}, shape={tuple(x.shape)}")

    def _iter_idx_from_csv(
        self,
        dataset_csv_path: str,
        n_token: int,
        max_samples: Optional[int],
        use_system_prompt: bool,
        batch_size: int,
        selected: Optional[torch.Tensor] = None,
    ) -> Generator[torch.Tensor, None, None]:
        """
        产出每个 batch 的激活 mask（GPU bool）。
        若 selected 非空，则只产出选中神经元列。
        """
        prompts = self._read_prompts_from_csv(dataset_csv_path, max_samples, use_system_prompt)

        # 直接用 fc1_module 自己的 weight/bias，避免 CPU numpy 来回搬运
        w = self.fc1_module.weight
        b = self.fc1_module.bias

        hidden_size = int(getattr(self.model_config, "hidden_size", w.shape[-1]))
        # 统一线性计算方向: output = x @ W_eff^T + b
        if w.shape[1] == hidden_size:
            w_eff = w
        elif w.shape[0] == hidden_size:
            w_eff = w.transpose(0, 1)
        else:
            raise ValueError(f"Unexpected fc1 weight shape={tuple(w.shape)}, hidden_size={hidden_size}")

        for prompt_batch in self._batch_iter(prompts, batch_size):
            hidden_2d, valid_mask_1d = self._capture_fc1_input_flat(prompt_batch, n_token)
            hidden_valid = hidden_2d[valid_mask_1d]

            # 全程 GPU pre-activation + mask
            if selected is not None:
                w_sel = w_eff.index_select(0, selected)
                b_sel = b.index_select(0, selected)
                preact = F.linear(hidden_valid, w_sel, b_sel)
            else:
                preact = F.linear(hidden_valid, w_eff, b)
            idx = preact > 0
            yield idx

    def buildOfflineRes(
        self,
        dataset_csv_path: str,
        n_token: int,
        max_samples: Optional[int] = 300,
        use_system_prompt: bool = False,
        batch_size: int = 8,
        max_neurons_for_clustering: Optional[int] = 4096,
    ) -> OfflineRes:
        # 先拿神经元数
        w = self.fc1_module.weight
        hidden_size = int(getattr(self.model_config, "hidden_size", w.shape[-1]))
        n_neuron = w.shape[0] if w.shape[1] == hidden_size else w.shape[1]

        freq_count = torch.zeros(n_neuron, dtype=torch.float64, device=self.fc1_device)
        total_tokens = 0

        # Pass-1: 全神经元激活频率（GPU）
        for idx in self._iter_idx_from_csv(
            dataset_csv_path, n_token, max_samples, use_system_prompt, batch_size, selected=None
        ):
            freq_count += idx.sum(dim=0, dtype=torch.float64)
            total_tokens += idx.shape[0]

        if total_tokens == 0:
            raise RuntimeError("No valid tokens processed.")

        freqs_t = freq_count / float(total_tokens)
        freqs = freqs_t.detach().cpu().numpy()
        tc_cc_ratio = neurons2TC_CCSorver.find_ratio_by_history_idx(freqs)

        # 神经元选择
        if max_neurons_for_clustering is None or n_neuron <= max_neurons_for_clustering:
            selected = torch.arange(n_neuron, device=self.fc1_device, dtype=torch.long)
        else:
            k = int(max_neurons_for_clustering)
            selected = torch.topk(freqs_t, k=k, dim=0).indices

        k_neuron = selected.numel()
        cw_acc = torch.zeros((k_neuron, k_neuron), dtype=torch.float32, device=self.fc1_device)

        # Pass-2: 子集 CW（GPU）
        for idx_sub in self._iter_idx_from_csv(
            dataset_csv_path, n_token, max_samples, use_system_prompt, batch_size, selected=selected
        ):
            x = idx_sub.to(torch.float32)
            cw_acc += x.transpose(0, 1) @ x

        cw_np = cw_acc.detach().cpu().numpy()
        hcns_sub, lcns_sub = neurons2GPUSorver.split_HCNS_LCNS_by_spectral(cw_np)

        selected_np = selected.detach().cpu().numpy()
        hcns = selected_np[hcns_sub]

        if selected.numel() == n_neuron:
            lcns = selected_np[lcns_sub]
        else:
            mask = np.ones(n_neuron, dtype=bool)
            mask[hcns] = False
            lcns = np.where(mask)[0]

        return OfflineRes(
            model_layer_config=self.model_layer_config,
            tc_cc_ratio=tc_cc_ratio,
            HCNS=np.asarray(hcns, dtype=np.int32),
            LCNS=np.asarray(lcns, dtype=np.int32),
        )


if __name__ == "__main__":
    model_layer_config = ModelLayerConfig(
        model_name="opt-30b",
        model_path="/mnt/sdb/llm_models/opt-30b",
        layer_idx=13,
    )

    model_bundle = ModelBundle(model_layer_config)
    solver = OfflineSolver(model_layer_config, model_bundle)

    dataset_csv_path = "/home/yswang/tvm_learn/cuda/DejaVu_TEMP/datasets/small_openorca.csv"

    offline_res = solver.buildOfflineRes(
        dataset_csv_path=dataset_csv_path,
        n_token=128,
        max_samples=300,
        use_system_prompt=False,
        batch_size=8,
        max_neurons_for_clustering=4096,
    )

    save_root = "/home/yswang/tvm_learn/cuda/DejaVu_TEMP"
    offline_res.save(save_root)

    loaded = OfflineRes.load(save_root, model_layer_config)
    print("tc_cc_ratio =", loaded.tc_cc_ratio)
    print("HCNS size =", loaded.HCNS.shape[0], "LCNS size =", loaded.LCNS.shape[0])