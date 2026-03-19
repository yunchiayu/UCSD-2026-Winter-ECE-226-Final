from dataclasses import dataclass
from pathlib import Path
import json
from typing import Any, Dict

@dataclass
class LlamaModelArgs:
    num_layers: int
    hidden_size: int
    num_heads: int
    num_kv_heads: int
    head_dim: int
    intermediate_size: int
    max_position_embeddings: int
    model_type: str
    num_experts: int
    num_experts_per_tok: int
    raw_config: dict

    @classmethod
    def init_from_config(cls, raw_config):
        # config_path = Path(model_config_path)
        # with config_path.open("r", encoding="utf-8") as handle:
        #     raw_config = json.load(handle)

        # Check hidden size is divisible by num_attention_heads
        hidden_size = raw_config["hidden_size"]
        num_heads = raw_config["num_attention_heads"]
        if hidden_size % num_heads != 0:
            raise ValueError(
                f"hidden_size {hidden_size} is not divisible by num_attention_heads {num_heads}, in config path: {config_path}"
            )

        config: Dict[str, Any] = {
            "num_layers": raw_config["num_hidden_layers"],
            "hidden_size": hidden_size,
            "num_heads": num_heads,
            "num_kv_heads": raw_config.get("num_key_value_heads", num_heads),
            "head_dim": raw_config.get("head_dim", hidden_size // num_heads),
            "intermediate_size": raw_config["intermediate_size"],
            "max_position_embeddings": raw_config["max_position_embeddings"],
            "model_type": raw_config["model_type"],
            "num_experts": raw_config.get("num_experts", 1),
            "num_experts_per_tok": raw_config.get("num_experts_per_tok", 1),
            "raw_config": raw_config,
        }
        
        return cls(**config)


class LlamaModel():
    def __init__(self, 
        model_args: LlamaModelArgs
    ):
        self.model_args = model_args
        self.operator_graph = None

        self.num_layers = model_args.num_layers
    
    def build_operator_graph(self,
        batch_size: int,
        sum_seq_len: int
    ):
        # Q, K, V, O
        B = batch_size
        D = self.model_args.hidden_size

        # QK, SV
        D_h = self.model_args.head_dim
        H_q = self.model_args.num_heads
        H_kv = self.model_args.num_kv_heads
        num_q_per_kv = H_q // H_kv
        Lin = sum_seq_len

        # FFN
        D_i = self.model_args.intermediate_size
        num_experts = self.model_args.num_experts
        num_experts_per_tok = self.model_args.num_experts_per_tok
        if batch_size * num_experts_per_tok < num_experts:
            effective_batch_size_per_expert = 1
        else:
            effective_batch_size_per_expert = (batch_size * num_experts_per_tok) / num_experts # Default: (16 * 8) / 64 = 2
        B_expert = int(effective_batch_size_per_expert)

        self.operator_graph = {
            "prefill":{
                "wq": { "type": "gemm", "params": { "B": 1, "M": B*Lin, "N": D, "K": D },},
                "wk": { "type": "gemm", "params": { "B": 1, "M": B*Lin, "N": H_kv * D_h, "K": D } },
                "wv": { "type": "gemm", "params": { "B": 1, "M": B*Lin, "N": H_kv * D_h, "K": D } },
                "wo": { "type": "gemm", "params": { "B": 1, "M": B*Lin, "N": D, "K": D } },
                "w1": { "type": "gemm", "params": { "B": 1, "M": B_expert*Lin, "N": D, "K": D_i } },
                "w3": { "type": "gemm", "params": { "B": 1, "M": B_expert*Lin, "N": D, "K": D_i } },
                "w2": { "type": "gemm", "params": { "B": 1, "M": B_expert*Lin, "N": D_i, "K": D } },
                "qK_softmax_sV_fused": {
                    "type": "qK_softmax_sV_fused",
                    "params": { "B": B, "H_kv": H_kv, "num_q_per_kv": num_q_per_kv, "Lin": Lin, "D_h": D_h },
                }
                # "qk": { "B": B * H_kv, "M": num_q_per_kv*Lin, "N": D_h, "K": Lin},
                # "sv": { "B": B * H_kv, "M": num_q_per_kv*Lin, "N": Lin, "K": D_h },
            },
            "decode":{
                "wq": { "type": "gemm", "params": { "B": 1, "M": B, "N": D, "K": D } },
                "wk": { "type": "gemm", "params": { "B": 1, "M": B, "N": H_kv * D_h, "K": D } },
                "wv": { "type": "gemm", "params": { "B": 1, "M": B, "N": H_kv * D_h, "K": D } },
                "wo": { "type": "gemm", "params": { "B": 1, "M": B, "N": D, "K": D } },
                "w1": { "type": "gemm", "params": { "B": 1, "M": B_expert, "N": D, "K": D_i } },
                "w3": { "type": "gemm", "params": { "B": 1, "M": B_expert, "N": D, "K": D_i } },
                "w2": { "type": "gemm", "params": { "B": 1, "M": B_expert, "N": D_i, "K": D } },
                "qK_softmax_sV_fused": {
                    "type": "qK_softmax_sV_fused",
                    "params": { "B": B, "H_kv": H_kv, "num_q_per_kv": num_q_per_kv, "Lin": Lin, "D_h": D_h },
                }
                # "qk": { "B": B * H_kv, "M": num_q_per_kv, "N": D_h, "K": Lin},
                # "sv": { "B": B * H_kv, "M": num_q_per_kv, "N": Lin, "K": D_h },
            },
        }
        return self.operator_graph
