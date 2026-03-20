class Evaluator:
    def __init__(self,
        hardware_config: dict,
        element_size: int,
    ):
        self.hardware_config = hardware_config
        self.element_size = element_size
    def evaluate_gemm(self, 
        B, M, N, K,
    ):  
        # hardware
        peak_bandwidth = self.hardware_config["GPU"]["peak_bandwidth"] # TB/s
        peak_compute = self.hardware_config["GPU"]["peak_compute"] # TFLOPS


        # GEMM parameters
        weight_size = B * (N * K) * self.element_size
        input_size = B * (M * N) * self.element_size
        output_size = B * (M * K) * self.element_size

        data_transfer_size = weight_size + input_size + output_size # bytes
        memory_time = data_transfer_size / peak_bandwidth * 1e-9 # ms

        # GEMM time
        total_flops = 2 * M * N * K
        compute_time = total_flops / peak_compute * 1e-9 # ms

        total_time = max(memory_time, compute_time)

        return total_time
    
    def evaluate_qK_softmax_sV_fused_decode(self,
        B, H_kv, num_q_per_kv, Lin, D_h,
    ):  
        # hardware
        peak_bandwidth = self.hardware_config["GPU"]["peak_bandwidth"] # TB/s
        peak_compute = self.hardware_config["GPU"]["peak_compute"] # TFLOPS

        # Data transfer size
        # B: (B * H_kv)
        # M: num_q_per_kv
        Q_size = (B * H_kv) * (num_q_per_kv) * D_h * self.element_size
        K_size = (B * H_kv) * Lin * D_h * self.element_size
        V_size = (B * H_kv) * Lin * D_h * self.element_size
        output_size = (B * H_kv) * (num_q_per_kv * Lin) * D_h * self.element_size
        data_transfer_size = Q_size + K_size + V_size + output_size # bytes
        memory_time = data_transfer_size / peak_bandwidth * 1e-9 # ms

        # QK_softmax_sV_fused time
        flops_QK = 2 * (B * H_kv) * num_q_per_kv * Lin * D_h
        flops_SV = 2 * (B * H_kv) * num_q_per_kv * Lin * D_h
        total_flops = flops_QK + flops_SV
        compute_time = total_flops / peak_compute * 1e-9 # ms

        total_time = max(memory_time, compute_time)

        return total_time, total_flops, data_transfer_size

    def evaluate_qK_softmax_sV_fused_prefill(self,
        B, H_kv, num_q_per_kv, Lin, D_h,
    ):  
        # hardware
        peak_bandwidth = self.hardware_config["GPU"]["peak_bandwidth"] # TB/s
        peak_compute = self.hardware_config["GPU"]["peak_compute"] # TF LOPS

        # Data transfer size
        # B: B * H_kv
        # M: num_q_per_kv * Lin
        # N: D_h
        # K: Lin

        # Data transfer size
        Q_size = (B * H_kv) * (num_q_per_kv * Lin) * D_h * self.element_size
        K_size = (B * H_kv) * Lin * D_h * self.element_size
        V_size = (B * H_kv) * Lin * D_h * self.element_size
        output_size = (B * H_kv) * (num_q_per_kv * Lin) * D_h * self.element_size
        data_transfer_size = Q_size + K_size + V_size + output_size # bytes
        memory_time = data_transfer_size / peak_bandwidth * 1e-9 # ms

        data_transfer_size_breakdown = {
            "Q": Q_size,
            "K": K_size,
            "V": V_size,
            "output": output_size,
        }

        # QK_softmax_sV_fused time
        flops_QK = 2 * (B * H_kv) * (num_q_per_kv * Lin) * D_h * Lin
        flops_SV = 2 * (B * H_kv) * (num_q_per_kv * Lin) * Lin * D_h
        total_flops = flops_QK + flops_SV
        compute_time = total_flops / peak_compute * 1e-9 # ms

        total_time = max(memory_time, compute_time)

        return total_time, total_flops, data_transfer_size, data_transfer_size_breakdown

    def evaluate_mamba_elementwise(self,
        B, D_i, L, N, op1, op2,
    ):
        # hardware
        peak_bandwidth = self.hardware_config["GPU"]["peak_bandwidth"] # TB/s
        peak_compute = self.hardware_config["GPU"]["peak_compute"] # TFLOPS

        # Dimension table
        dim_table = {"B": B, "D_i": D_i, "L": L, "N": N}

        # Data transfer size
        op1_size = self.element_size
        op2_size = self.element_size
        for dim_name in op1:
            dim_value = dim_table[dim_name]
            op1_size *= dim_value
        for dim_name in op2:
            dim_value = dim_table[dim_name]
            op2_size *= dim_value
        output_size = B * D_i * L * N * self.element_size
        data_transfer_size = op1_size + op2_size + output_size # bytes
        memory_time = data_transfer_size / peak_bandwidth * 1e-9 # ms

        # Elementwise time
        total_flops = B * D_i * L * N
        compute_time = total_flops / peak_compute * 1e-9 # ms

        total_time = max(memory_time, compute_time)

        return total_time

    def evaluate_conv1d(self,
        B, L, Cin, Cout, K, G
    ):
        # hardware
        peak_bandwidth = self.hardware_config["GPU"]["peak_bandwidth"] # TB/s
        peak_compute = self.hardware_config["GPU"]["peak_compute"] # TFLOPS

        # Data transfer size
        input_size = B * L * Cin * self.element_size
        output_size = B * L * Cout * self.element_size
        weight_size = (Cin / G) * Cout * K * self.element_size
        data_transfer_size = input_size + output_size + weight_size # bytes
        memory_time = data_transfer_size / peak_bandwidth * 1e-9 # ms

        # Conv1d time
        total_flops = 2 * B * L * K * (Cin/G) * Cout
        compute_time = total_flops / peak_compute * 1e-9 # ms

        total_time = max(memory_time, compute_time)

        return total_time
        


    def evaluate_single_layer(self, model_operator: dict):
        result_dict = {
            "unit": {
              "time": "ms",
              "flops": "GFLOPs",
              "throughput": "TFLOPS",
              "arithmetic_intensity": "FLOPS/byte",
            },
            "prefill": {
                "overall": {},
                "breakdown": {}
            },
            "decode": {
                "overall": {},
                "breakdown": {}
            }
        }
        # prefill
        prefill_operator = model_operator["prefill"]
        prefill_time = 0.0 # ns
        prefill_flops = 0.0 # FLOPs
        prefill_data_transfer_size = 0.0 
        for op_name, op_data in prefill_operator.items():
            if op_data["type"] == "gemm":
                B, M, N, K = op_data["params"]["B"], op_data["params"]["M"], op_data["params"]["N"], op_data["params"]["K"]
                operation_time = self.evaluate_gemm(B, M, N, K)
                num_flops = 2 * M * N * K
                throughput = num_flops / operation_time * 1e-9 # TFLOPS/s

                weight_size = B * (N * K) * self.element_size
                input_size = B * (M * N) * self.element_size
                output_size = B * (M * K) * self.element_size
                data_transfer_size = weight_size + input_size + output_size # bytes
                arithmetic_intensity = num_flops / data_transfer_size # FLOPS/byte
                metadata = {
                    "params": {
                        "B": B,
                        "M": M,
                        "N": N,
                        "K": K,
                    },
                }
            elif op_data["type"] == "qK_softmax_sV_fused":
                B, H_kv, num_q_per_kv, Lin, D_h = op_data["params"]["B"], op_data["params"]["H_kv"], op_data["params"]["num_q_per_kv"], op_data["params"]["Lin"], op_data["params"]["D_h"]
                operation_time, num_flops, data_transfer_size, data_transfer_size_breakdown = self.evaluate_qK_softmax_sV_fused_prefill(B, H_kv, num_q_per_kv, Lin, D_h)
                throughput = num_flops / operation_time * 1e-9 # TFLOPS/s
                arithmetic_intensity = num_flops / data_transfer_size # FLOPS/byte
                metadata = {
                    "params": {
                        "B": B,
                        "H_kv": H_kv,
                        "num_q_per_kv": num_q_per_kv,
                        "Lin": Lin,
                        "D_h": D_h,
                    },
                    "flops": num_flops,
                    "data_transfer_size": data_transfer_size,
                    "data_transfer_size_breakdown": data_transfer_size_breakdown,
                }
            elif op_data["type"] == "mamba_elementwise":
                B, D_i, L, N, op1, op2 = op_data["params"]["B"], op_data["params"]["D_i"], op_data["params"]["L"], op_data["params"]["N"], op_data["params"]["op1"], op_data["params"]["op2"]
                operation_time = self.evaluate_mamba_elementwise(B, D_i, L, N, op1, op2)
                num_flops = B * D_i * L * N
                throughput = num_flops / operation_time * 1e-9 # TFLOPS/s
                arithmetic_intensity = num_flops / data_transfer_size # FLOPS/byte
                metadata = {
                    "params": {
                        "B": B,
                        "D_i": D_i,
                        "L": L,
                        "N": N,
                        "op1": op1,
                        "op2": op2,
                    },
                }
            elif op_data["type"] == "conv1d":
                B, L, Cin, Cout, K, G = op_data["params"]["B"], op_data["params"]["L"], op_data["params"]["Cin"], op_data["params"]["Cout"], op_data["params"]["K"], op_data["params"]["G"]
                operation_time = self.evaluate_conv1d(B, L, Cin, Cout, K, G)
                num_flops = 2 * B * L * K * (Cin/G) * Cout
                throughput = num_flops / operation_time * 1e-9 # TFLOPS/s
                arithmetic_intensity = num_flops / data_transfer_size # FLOPS/byte
                metadata = {
                    "params": {
                        "B": B,
                        "L": L,
                        "Cin": Cin,
                        "Cout": Cout,
                        "K": K,
                        "G": G,
                    },
                }
            else:
                raise ValueError(f"Unsupported operator: {op_data['type']}")
            prefill_time += operation_time
            prefill_flops += num_flops
            prefill_data_transfer_size += data_transfer_size
            result_dict["prefill"]["breakdown"][op_name] = {#operation_time # ms
                "type": op_data["type"],
                "time": operation_time,
                "flops": num_flops * 1e-9, # GFLOPs
                "throughput": throughput,
                "arithmetic_intensity": arithmetic_intensity,
                "metadata": metadata,
            }
        prefill_throughput = prefill_flops / prefill_time * 1e-9 # TFLOPS/s
        prefill_arithmetic_intensity = prefill_flops / prefill_data_transfer_size # FLOPS/byte
        result_dict["prefill"]["overall"] = {
            "time": prefill_time,
            "flops": prefill_flops * 1e-9, # GFLOPs
            "throughput": prefill_throughput,
            "arithmetic_intensity": prefill_arithmetic_intensity,
        }

        # decode
        decode_operator = model_operator["decode"]
        decode_time = 0.0
        decode_flops = 0.0
        decode_data_transfer_size = 0.0
        for op_name, op_data in decode_operator.items():
            if op_data["type"] == "gemm":
                B, M, N, K = op_data["params"]["B"], op_data["params"]["M"], op_data["params"]["N"], op_data["params"]["K"]
                operation_time = self.evaluate_gemm(B, M, N, K)
                num_flops = 2 * M * N * K
                throughput = num_flops / operation_time * 1e-9 # TFLOPS/s

                weight_size = B * (N * K) * self.element_size
                input_size = B * (M * N) * self.element_size
                output_size = B * (M * K) * self.element_size
                data_transfer_size = weight_size + input_size + output_size # bytes
                arithmetic_intensity = num_flops / data_transfer_size # FLOPS/byte
                metadata = {
                    "params": {
                        "B": B,
                        "M": M,
                        "N": N,
                        "K": K,
                    },
                }
            elif op_data["type"] == "qK_softmax_sV_fused":
                B, H_kv, num_q_per_kv, Lin, D_h = op_data["params"]["B"], op_data["params"]["H_kv"], op_data["params"]["num_q_per_kv"], op_data["params"]["Lin"], op_data["params"]["D_h"]
                operation_time, num_flops, data_transfer_size = self.evaluate_qK_softmax_sV_fused_decode(B, H_kv, num_q_per_kv, Lin, D_h)
                throughput = num_flops / operation_time * 1e-9 # TFLOPS/s
                arithmetic_intensity = num_flops / data_transfer_size # FLOPS/byte

                metadata = {
                    "params": {
                        "B": B,
                        "H_kv": H_kv,
                        "num_q_per_kv": num_q_per_kv,
                        "Lin": Lin,
                        "D_h": D_h,
                    },
                }
            decode_time += operation_time
            decode_flops += num_flops
            decode_data_transfer_size += data_transfer_size
            result_dict["decode"]["breakdown"][op_name] = {#operation_time # ms
                "time": operation_time,
                "type": op_data["type"],
                "flops": num_flops * 1e-9, # GFLOPs
                "throughput": throughput,
                "arithmetic_intensity": arithmetic_intensity,
                "metadata": metadata,
            }
        decode_throughput = decode_flops / decode_time * 1e-9 # TFLOPS/s
        decode_arithmetic_intensity = decode_flops / decode_data_transfer_size # FLOPS/byte
        result_dict["decode"]["overall"] = {
            "time": decode_time,
            "flops": decode_flops * 1e-9, # GFLOPs
            "throughput": decode_throughput,
            "arithmetic_intensity": decode_arithmetic_intensity,
        }

        return result_dict
    
    def evaluate_model(self, 
        model, 
        batch_size,
        sum_seq_len,
        gen_seq_len,
    ):
        model_operator_graph = model.build_operator_graph(batch_size, sum_seq_len)
        performance_per_layer = self.evaluate_single_layer(model_operator_graph)

        num_layers = model.num_layers
        prefill_time = num_layers * performance_per_layer["prefill"]["overall"]["time"] # ms
        decode_time = num_layers * performance_per_layer["decode"]["overall"]["time"] * gen_seq_len # ms


        decoding_throughput = batch_size * gen_seq_len / decode_time * 1e3 # tokens/s



        performace = {
            "unit":{
                "time": "ms",
                "throughput": "tokens/s",
            },
            "TTFT_time": prefill_time,
            "decoding_time": decode_time,
            "decoding_throughput": decoding_throughput,
            "performance_per_layer": performance_per_layer
        }

        return performace


        

