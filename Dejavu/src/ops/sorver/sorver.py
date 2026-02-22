因为冷热神经元也即FFN的两层权重基本是相对稳定的，也就是不随我输入X的变化而波动很大

所以第一层求解器就是neurons2GPUSorver，也即离线统计不同X输入时候Predictor得到的M

然后用M计算共激活矩阵CW，然后用M和CW得到neuron2GPU的划分以及neuron2TC-CC的划分，也即神经元的划分都是离线做好的

from DejaVu.Dejavu.src.ops.sorver.SystemCommProfiler import SystemCommProfiler
from dataclasses import dataclass
import os
from typing import Optional
from transformers import GPT2Config
import numpy as np
from sklearn.cluster import SpectralClustering
from dataclasses import asdict
import json

@dataclass 
class ModelLayerConfig:
    # 也即哪个模型的哪个层的xxxx配置
    model_config : GPT2Config
    layer_idx : int


@dataclass
class OfflineRes:
    model_layer_config: ModelLayerConfig
    tc_cc_ratio: float
    HCNS: np.ndarray
    LCNS: np.ndarray
    
    @staticmethod
    def load(save_root: str, model_layer_config: ModelLayerConfig):

        model_name = model_layer_config.model_config._name_or_path
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

        model_name = self.model_layer_config.model_config._name_or_path
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
    
    
class X2ChunkSorver:
    def __init__(self):
        self.comm_cost = SystemCommProfiler
        # self.flashffn_cost = xxxx
    
    
class OfflineSolver:
    def __init__(self, model_layer_config):
        self.model_layer_config = model_layer_config
        self.idx_list = []
        self.CW_list = []
        self.offline_res = OfflineRes(
            model_layer_config=model_layer_config,
            tc_cc_ratio=None,
            HCNS=None,
            LCNS=None
        )
    
    def build_idx_list(self):
            # 根据model_layer_config不断run DeJaVu然后保存不同的idx
            idx_list = []
            
            self.idx_list = idx_list
        
    def idx2CW(self):
        """
        计算每个 idx 对应的共激活矩阵 CW
        idx_list: list of idx 矩阵 (每个为 neuron × neuron 二进制或激活 mask)
        return: list of CW 矩阵
        """
        self.CW_list = [idx.T @ idx for idx in self.idx_list]

    def buildOfflineRes(self, model_layer_config):
        model_layer_config = ModelLayerConfig(
            model_config=model_layer_config.model_config,
            layer_idx=model_layer_config.layer_idx
        )
        self.build_idx_list()
        idx_list = self.idx_list 
        self.idx2CW(idx_list)
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
    from transformers import GPT2Config

    # 1. 构造模型/层信息
    config = GPT2Config.from_pretrained("facebook/opt-1.3b")
    model_layer_config = ModelLayerConfig(
        model_config=config,
        layer_idx=13
    )

    # 2. 离线求解
    solver = OfflineSolver(model_layer_config)
    solver.buildOfflineRes()

    # 3. 指定保存根路径
    SAVE_ROOT = "/home/yswang/tvm_learn/cuda/DejaVu_TEMP"

    # 4. 保存
    solver.offline_res.save(SAVE_ROOT)

    # 5. 测试读取
    loaded = OfflineRes.load(SAVE_ROOT, model_layer_config)
    print("tc_cc_ratio =", loaded.tc_cc_ratio)