# 因为冷热神经元也即FFN的两层权重基本是相对稳定的，也就是不随我输入X的变化而波动很大

# 所以第一层求解器就是neurons2GPUSorver，也即离线统计不同X输入时候Predictor得到的M

# 然后用M计算共激活矩阵CW，然后用M和CW得到neuron2GPU的划分以及neuron2TC-CC的划分，也即神经元的划分都是离线做好的

# from DejaVu.Dejavu.src.ops.sorver.SystemCommProfiler import SystemCommProfiler
from dataclasses import dataclass, field
import os
import torch
from typing import Optional
from transformers import GPT2Config
import numpy as np
from sklearn.cluster import SpectralClustering
from dataclasses import asdict
import json
from typing import List, Union
from transformers import (
    AutoConfig,                 # 加载/推断模型配置
    AutoTokenizer,              # 自动加载 tokenizer
    AutoModelForCausalLM,        # 自动加载因果 LM（GPT/OPT/LLaMA）
    PretrainedConfig,
    PreTrainedModel,
    PreTrainedTokenizerBase
)

@dataclass
class ModelLayerConfig:
    model_name: str      # 用于命名和输出
    model_path: str      # 用于加载模型
    layer_idx: int

    

class ModelBundle:
    def __init__(self, cfg: ModelLayerConfig):
        self.model_config = AutoConfig.from_pretrained(
            cfg.model_path,
            torch_dtype=torch.float16
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            cfg.model_path,
            config=self.model_config,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        self.tokenizer = AutoTokenizer.from_pretrained(cfg.model_path)

@dataclass
class X:
    prompt: str
    n_token: int
    model_layer_config: ModelLayerConfig
    hidden_matrix: np.ndarray = field(init=False, default=None)
    input_ids: np.ndarray = field(init=False, default=None)
    attention_mask: np.ndarray = field(init=False, default=None)
    seq_len: int = field(init=False, default=0)
   

@dataclass
class OfflineRes:
    model_layer_config: ModelLayerConfig
    tc_cc_ratio: float
    HCNS: np.ndarray
    LCNS: np.ndarray
    
    @staticmethod
    def load(save_root: str, model_layer_config: ModelLayerConfig):

        model_name = model_layer_config.model_name
        layer_idx = model_layer_config.layer_idx

        save_dir = os.path.join(save_root, model_name, f"layer_{layer_idx}")

        with open(os.path.join(save_dir, "meta.json"), "r") as f:
            meta = json.load(f)
        HCNS = np.load(os.path.join(save_dir, "HCNS.npy"))
        LCNS = np.load(os.path.join(save_dir, "LCNS.npy"))

        return OfflineRes(
            model_layer_config=model_layer_config,
            tc_cc_ratio=meta["tc_cc_ratio"],
            HCNS=HCNS,
            LCNS=LCNS,
        )

    def save(self, save_root: str):

        model_name = self.model_layer_config.model_name
        layer_idx = self.model_layer_config.layer_idx

        save_dir = os.path.join(save_root, model_name, f"layer_{layer_idx}")
        os.makedirs(save_dir, exist_ok=True)

        meta = {
            "model_name": model_name,
            "layer_idx": layer_idx,
            "tc_cc_ratio": self.tc_cc_ratio,
        }

        with open(os.path.join(save_dir, "meta.json"), "w") as f:
            json.dump(meta, f, indent=4)

        np.save(os.path.join(save_dir, "HCNS.npy"), self.HCNS)
        np.save(os.path.join(save_dir, "LCNS.npy"), self.LCNS)

        print(f"[OfflineRes] Saved to {save_dir}")

class neurons2GPUSorver:
    def __init__(self):
        pass
    
    def split_HCNS_LCNS_by_spectral(self, CW_list):
        # 1) 聚合共激活矩阵（求平均）
        CW_stack = np.stack(CW_list, axis=0)            # shape: (k, n, n)
        CW_mean  = CW_stack.mean(axis=0)               # shape: (n, n)
        
        # 2) 谱聚类，使用 precomputed affinity 矩阵
        sc = SpectralClustering(
            n_clusters=2,
            affinity='precomputed',   # 把 CW_mean 当作亲和力 / 相似矩阵
            assign_labels='kmeans',   # 用 KMeans 对嵌入结果分簇（默认）
            random_state=42
        )
        labels = sc.fit_predict(CW_mean)               # shape: (n,)
        
        # 3) 判断哪一类是“高共激活”
        # 简单用每类的平均共激活度比较
        coact_scores = CW_mean.sum(axis=1)             # 每个神经元的总体共激活度
        mean0 = coact_scores[labels == 0].mean()
        mean1 = coact_scores[labels == 1].mean()
        
        if mean0 > mean1:
            HCNS = np.where(labels == 0)[0]
            LCNS = np.where(labels == 1)[0]
        else:
            HCNS = np.where(labels == 1)[0]
            LCNS = np.where(labels == 0)[0]
        
        return HCNS, LCNS
        
# 根据离线的一堆idx统计得到一个大致TC-CC的划分阈值(激活频率)，然后结合现在输出的M统计HANS和LANS
class neurons2TC_CCSorver:
    def __init__(self):
        pass
    
    def activation_freq(self, idx_list):
        idx_stack = np.stack(idx_list, axis=0) # (k, n_neurons)
        return idx_stack.sum(axis=0) / len(idx_list)
    
    def find_ratio_by_history_idx(self, idx_list):
        # 离线预测，使得TC和CC的计算开销均衡，以来取代目前的naive版本
        freqs = self.activation_freq(idx_list)
        ratio = float(np.mean(freqs))
        return ratio
    
    
# class X2ChunkSorver:
#     def __init__(self):
#         self.comm_cost = SystemCommProfiler
#         # self.flashffn_cost = xxxx
    
    
class OfflineSolver:
    def __init__(self, model_layer_config, model_boundle):
        self.model_layer_config = model_layer_config
        self.model = model_boundle.model
        self.tokenizer = model_boundle.tokenizer
        self.idx_list = []
        self.x_list = []
        self.CW_list = []
        self.offline_res = OfflineRes(
            model_layer_config=model_layer_config,
            tc_cc_ratio=None,
            HCNS=None,
            LCNS=None
        )
        self.fc1 = None
        self.b1 = None
        
    def load_fc1_weight(self):
        layer_idx = self.model_layer_config.layer_idx

        # 方案A：优先从统一 remap 后的 state_dict 读取（最稳）
        sd = self.model.state_dict()
        k_w = f"transformer.layers.{layer_idx}.mlp.fc1.weight"
        k_b = f"transformer.layers.{layer_idx}.mlp.fc1.bias"
        if k_w in sd:
            self.fc1 = sd[k_w].detach().cpu().numpy()
            self.b1 = sd[k_b].detach().cpu().numpy()
            return

        # 方案B：兼容 HF 原生结构
        if hasattr(self.model, "transformer") and hasattr(self.model.transformer, "h"):  # GPT2
            layer = self.model.transformer.h[layer_idx]
            self.fc1 = layer.mlp.c_fc.weight.detach().cpu().numpy()
            self.b1 = layer.mlp.c_fc.bias.detach().cpu().numpy()
            return

        if hasattr(self.model, "model") and hasattr(self.model.model, "decoder"):  # OPT
            layer = self.model.model.decoder.layers[layer_idx]
            self.fc1 = layer.fc1.weight.detach().cpu().numpy()
            self.b1 = layer.fc1.bias.detach().cpu().numpy()
            return

        raise KeyError(f"Cannot locate fc1 for layer {layer_idx}")
    
    def extract_hidden_matrix(
            self,
            x: X,
            token_select: str = "all_valid",  # all_valid | last_valid | all_with_pad
        ) -> np.ndarray:
            """
            生成 x.hidden_matrix
            """
            tokenizer = self.tokenizer
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token

            inputs = tokenizer(
                x.prompt,
                padding="max_length",
                truncation=True,
                max_length=x.n_token,
                return_tensors="pt",
            )

            # 保存 token 信息
            x.input_ids = inputs["input_ids"][0].detach().cpu().numpy()
            x.attention_mask = inputs["attention_mask"][0].detach().cpu().numpy()
            x.seq_len = int(x.attention_mask.sum())

            # 找到 fc1
            layer_idx = self.model_layer_config.layer_idx
            fc1 = self._resolve_fc1_module(self.model, layer_idx)

            captured = {"x": None}

            def _pre_hook(module, module_inputs):
                captured["x"] = module_inputs[0].detach()

            handle = fc1.register_forward_pre_hook(_pre_hook)
            try:
                with torch.no_grad():
                    _ = self.model(**inputs, return_dict=True)
            finally:
                handle.remove()

            if captured["x"] is None:
                raise RuntimeError("Failed to capture FFN input.")

            hidden = captured["x"][0]  # (S, H)
            if token_select == "all_valid":
                mat = hidden[: x.seq_len, :]
            elif token_select == "last_valid":
                mat = hidden[x.seq_len - 1 : x.seq_len, :]
            elif token_select == "all_with_pad":
                mat = hidden
            else:
                raise ValueError(f"Unknown token_select={token_select}")

            x.hidden_matrix = mat.cpu().numpy()
            return x.hidden_matrix
          
        
    def build_x_list(
        self,
        dataset_csv_path: str,
        n_token: int,
        max_samples: Optional[int] = None,
        use_system_prompt: bool = False,
    ) -> list:
        """
        从 OpenOrca CSV 构造 x_list: list[X]
        - 默认使用 question 作为 prompt
        - use_system_prompt=True 时，拼接 system_prompt + question
        """
        import csv
        x_list: list[X] = []

        with open(dataset_csv_path, "r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)

            for row in reader:
                question = (row.get("question") or "").strip()
                if not question:
                    continue

                if use_system_prompt:
                    system_prompt = (row.get("system_prompt") or "").strip()
                    prompt = (
                        f"{system_prompt}\n\n{question}"
                        if system_prompt
                        else question
                    )
                else:
                    prompt = question

                # 构造 X 对象
                x = X(
                    prompt=prompt,
                    n_token=n_token,
                    model_layer_config=self.model_layer_config,
                )

                # 现 提取 hidden_matrix
                self.extract_hidden_matrix(x)

                x_list.append(x)

                if max_samples is not None and len(x_list) >= max_samples:
                    break

        return x_list

    def build_idx_list(self):
        """
        x_list 可为:
        - list[X]: 每个元素用 x.hidden_matrix
        - list[np.ndarray]: 每个元素 shape (n_token, hidden_dim) 或 (hidden_dim,)
        返回 idx_list: 每个元素 shape (n_token, n_neuron)
        """
        self.load_fc1_weight()
        W = self.fc1  # (n_neuron, hidden_dim)
        b = self.b1   # (n_neuron,)
        idx_list = []

        for item in self.x_list:
            Xmat = item.hidden_matrix if isinstance(item, X) else item
            if Xmat is None:
                raise ValueError("Found empty hidden_matrix in x_list.")
            if Xmat.ndim == 1:
                Xmat = Xmat[None, :]  # (1, hidden_dim)

            preact = Xmat @ W.T + b
            idx = (preact > 0).astype(np.int8)
            idx_list.append(idx)
        self.idx_list = idx_list
        
    def idx2CW(self):
        """
        计算每个 idx 对应的共激活矩阵 CW
        idx_list: list of idx 矩阵 (每个为 neuron × neuron 二进制或激活 mask)
        return: list of CW 矩阵
        """
        self.CW_list = [idx.T @ idx for idx in self.idx_list]

    def buildOfflineRes(self):
        self.build_idx_list()
        idx_list = self.idx_list 
        self.idx2CW()
        CW_list = self.CW_list
        neurons_2GPU_sorver = neurons2GPUSorver()
        neurons_2TC_CC_sorver =neurons2TC_CCSorver()

        HCNS, LCNS = neurons_2GPU_sorver.split_HCNS_LCNS_by_spectral(CW_list)
        tc_cc_ratio = neurons_2TC_CC_sorver.find_ratio_by_history_idx(idx_list)
        self.offline_res = OfflineRes(
            model_layer_config=model_layer_config,
            tc_cc_ratio=tc_cc_ratio,
            HCNS=HCNS,
            LCNS=LCNS
        )
    

if __name__ == "__main__":

    # 1. 构造模型/层信息
    model_layer_config = ModelLayerConfig(
        model_name="opt-30b",
        model_path="/mnt/sdb/llm_models/opt-30b",
        layer_idx=13
    )
    
    # 2. 加载模型 bundle
    model_bundle = ModelBundle(model_layer_config)

    # 正确访问 config
    total_layers = model_bundle.model_config.num_hidden_layers
    hidden_dim   = model_bundle.model_config.hidden_size
    print(f"layers: {total_layers}, hidden dim: {hidden_dim}")

    # 3. 离线求解
    solver = OfflineSolver(model_layer_config, model_bundle)

    # 3a) 构造 x_list （需要你自己准备一个合法的 CSV 数据集）
    x_list = solver.build_x_list(
        dataset_csv_path="/home/yswang/tvm_learn/cuda/DejaVu_TEMP/datasets/small_openorca.csv",
        n_token=128,
        max_samples=300
    )

    # 3b) 计算 idx_list 和 CW_list
    solver.build_idx_list(x_list)
    solver.idx2CW()

    # 3c) 构建 OfflineRes
    solver.buildOfflineRes()

    # 4. 指定保存根路径
    SAVE_ROOT = "/home/yswang/tvm_learn/cuda/DejaVu_TEMP"

    # 5. 保存
    solver.offline_res.save(SAVE_ROOT)

    # 6. 测试读取
    loaded = OfflineRes.load(SAVE_ROOT, model_layer_config)
    print("tc_cc_ratio =", loaded.tc_cc_ratio)