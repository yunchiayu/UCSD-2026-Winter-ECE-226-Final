from dataclasses import dataclass
from pathlib import Path
import json
from typing import Any, Dict

@dataclass
class MambaModelArgs:
    num_layers: int
    conv_kernel: int
    hidden_size: int
    intermediate_size: int
    state_size: int
    time_step_rank: int
    model_type: str
    raw_config: dict

    @classmethod
    def init_from_config(cls, raw_config):

        config: Dict[str, Any] = {
            "num_layers": raw_config["num_hidden_layers"],
            "conv_kernel": raw_config["conv_kernel"],
            "hidden_size": raw_config["hidden_size"],
            "intermediate_size": raw_config["intermediate_size"],
            "state_size": raw_config["state_size"],
            "time_step_rank": raw_config["time_step_rank"],
            "model_type": raw_config["model_type"],
            "raw_config": raw_config,
        }
        
        return cls(**config)


class MambaModel():
    def __init__(self, 
        model_args: MambaModelArgs
    ):
        self.model_args = model_args
        self.operator_graph = None

        self.num_layers = model_args.num_layers
    
    def build_operator_graph(self,
        batch_size: int,
        sum_seq_len: int
    ):
        # ===== Parameters =====
        B = batch_size
        D = self.model_args.hidden_size
        D_i = self.model_args.intermediate_size
        N = self.model_args.state_size
        P = self.model_args.time_step_rank
        K = self.model_args.conv_kernel
        Lin = sum_seq_len

        self.operator_graph = {
            # ===== Skipped Operators =====
            # "softplus": { "type": "softplus", "params": {} },
            # "update_ssm_state": { "type": "update_ssm_state", "params": {} }, (minmal latency)
            # =============================
            "prefill":{
                "in_proj": { "type": "gemm", "params": { "B": 1, "M": B*Lin, "N": D, "K": D_i * 2 } }, # Linear (B, L, D) -> (B, L, D_i * 2)
                "conv1d": { "type": "conv1d", "params": { "B": B, "L": Lin, "Cin": D_i, "Cout": D_i, "K": K, "G": D_i } },
                "x_proj": { "type": "gemm", "params": { "B": 1, "M": B*Lin, "N": D_i, "K": P + 2*N } }, # Linear (B, L, D_i) -> (B, L, P + 2*N)
                "dt_proj": { "type": "gemm", "params": { "B": 1, "M": B*Lin, "N": P, "K": D_i } }, # Linear (B, L, P) -> (B, L, D_i)
                "discrete_A": { "type": "mamba_elementwise", "params": { "B": B, "D_i": D_i, "L": Lin, "N": N, "op1": ["B", "D_i", "L"], "op2": ["D_i", "N"] }},
                "discrete_B": { "type": "mamba_elementwise", "params": { "B": B, "D_i": D_i, "L": Lin, "N": N, "op1": ["B", "D_i", "L"], "op2": ["B", "L", "N"] }},
                "deltaB_u": { "type": "mamba_elementwise", "params": { "B": B, "D_i": D_i, "L": Lin, "N": N, "op1": ["B", "D_i", "L", "N"], "op2": ["B", "D_i", "L"] }},
                # =============================
                "scan_ssm_update": { "type": "mamba_elementwise", "params": { "B": B, "D_i": D_i, "L": 1, "N": N, "op1": ["B", "D_i", "N"], "op2": ["B", "D_i", "N"] }}, # Lin == 1, since it use for loop
                "scan_matmul":  { "type": "gemm", "params": { "B": B, "M": D_i, "N": N, "K": 1 } }, # GEMV (B, D_i, N) x (B, N, 1) -> (B, D_i, 1)
                # ------------------------------------------------------------
                # "scan_ssm_update": { "type": "mamba_elementwise", "params": { "B": B*Lin, "D_i": D_i, "L": 1, "N": N, "op1": ["B", "D_i", "N"], "op2": ["B", "D_i", "N"] }}, # Lin == 1, since it use for loop
                # "scan_matmul":  { "type": "gemm", "params": { "B": B*Lin, "M": D_i, "N": N, "K": 1 } }, # GEMV (B, D_i, N) x (B, N, 1) -> (B, D_i, 1)
                # =============================
                "scan_add_D": {"type": "mamba_elementwise", "params": { "B": B, "D_i": D_i, "L": 1, "N": N, "op1": ["B", "D_i", "L"], "op2": ["D_i"] }}, # Lin == 1, since output (y) dont have Lin dimenstion
                "scan_act_gate": {"type": "mamba_elementwise", "params": { "B": B, "D_i": D_i, "L": 1, "N": N, "op1": ["B", "D_i", "L"], "op2": ["B", "D_i", "L"] }}, # Lin == 1, since output (y) dont have Lin dimenstion
                "out_proj": { "type": "gemm", "params": { "B": 1, "M": B*Lin, "N": D_i, "K": D } }, # Linear (B, L, D_i) -> (B, L, D)
            },
            "decode":{
                "in_proj": { "type": "gemm", "params": { "B": 1, "M": B, "N": D, "K": D_i * 2 } }, # Linear (B, L, D) -> (B, L, D_i * 2)
                "x_proj": { "type": "gemm", "params": { "B": 1, "M": B, "N": D_i, "K": P + 2*N } }, # Linear (B, L, D_i) -> (B, L, P + 2*N)
                "dt_proj": { "type": "gemm", "params": { "B": 1, "M": B, "N": P, "K": D_i } }, # Linear (B, L, P) -> (B, L, D_i)
                "discrete_A": { "type": "mamba_elementwise", "params": { "B": B, "D_i": D_i, "L": 1, "N": N, "op1": ["B", "D_i", "L"], "op2": ["D_i", "N"] }},
                "discrete_B": { "type": "mamba_elementwise", "params": { "B": B, "D_i": D_i, "L": 1, "N": N, "op1": ["B", "D_i", "L"], "op2": ["B", "L", "N"] }},
                "deltaB_u": { "type": "mamba_elementwise", "params": { "B": B, "D_i": D_i, "L": 1, "N": N, "op1": ["B", "D_i", "L", "N"], "op2": ["B", "D_i", "L"] }},
                "scan_ssm_update": { "type": "mamba_elementwise", "params": { "B": B, "D_i": D_i, "L": 1, "N": N, "op1": ["B", "D_i", "N"], "op2": ["B", "D_i", "N"] }},
                "scan_matmul":  { "type": "gemm", "params": { "B": B, "M": D_i, "N": N, "K": 1 } }, # GEMV (B, D_i, N) x (B, N, 1) -> (B, D_i, 1)
                "scan_add_D": {"type": "mamba_elementwise", "params": { "B": B, "D_i": D_i, "L": 1, "N": N, "op1": ["B", "D_i", "L"], "op2": ["D_i"] }}, 
                "scan_act_gate": {"type": "mamba_elementwise", "params": { "B": B, "D_i": D_i, "L": 1, "N": N, "op1": ["B", "D_i", "L"], "op2": ["B", "D_i", "L"] }}, 
                "out_proj": { "type": "gemm", "params": { "B": 1, "M": B, "N": D_i, "K": D } }, # Linear (B, L, D_i) -> (B, L, D)
            },
        }
        return self.operator_graph
