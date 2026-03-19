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
        gemm_time = total_flops / peak_compute * 1e-9 # ms

        total_time = max(memory_time, gemm_time)

        return total_time
    
    def evaluate_qK_softmax_sV_fused(self,
        B, H_kv, num_q_per_kv, Lin, D_h,
    ):  
        # hardware
        peak_bandwidth = self.hardware_config["GPU"]["peak_bandwidth"] # TB/s
        peak_compute = self.hardware_config["GPU"]["peak_compute"] # TFLOPS

        # Data transfer size
        KV_cache_size = 2 * B * H_kv * Lin * D_h * self.element_size # 2 for K & V
        input_size = B * num_q_per_kv * H_kv * D_h
        output_size = B * num_q_per_kv * H_kv * D_h
        data_transfer_size = KV_cache_size + input_size + output_size # bytes
        memory_time = data_transfer_size / peak_bandwidth * 1e-9 # ms

        # QK_softmax_sV_fused time
        total_flops = 2 * (B * H_kv) * num_q_per_kv * Lin * D_h
        qK_softmax_sV_fused_time = total_flops / peak_compute * 1e-9 # ms

        total_time = max(memory_time, qK_softmax_sV_fused_time)

        return total_time

    def evaluate_single_layer(self, model_operator: dict):
        result_dict = {
            "prefill": {
                "total_time": 0.0,
                "breakdown": {
                    "unit": "ms",
                }
            },
            "decode": {
                "total_time": 0.0,
                "breakdown": {
                    "unit": "ms",
                }
            }
        }
        # prefill
        prefill_operator = model_operator["prefill"]
        prefill_time = 0.0
        prefill_flops = 0.0
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
            elif op_data["type"] == "qK_softmax_sV_fused":
                B, H_kv, num_q_per_kv, Lin, D_h = op_data["params"]["B"], op_data["params"]["H_kv"], op_data["params"]["num_q_per_kv"], op_data["params"]["Lin"], op_data["params"]["D_h"]
                operation_time = self.evaluate_qK_softmax_sV_fused(B, H_kv, num_q_per_kv, Lin, D_h)
                num_flops = 2 * (B * H_kv) * num_q_per_kv * Lin * D_h
                throughput = num_flops / operation_time * 1e-9 # TFLOPS/s

                KV_cache_size = 2 * B * H_kv * Lin * D_h * self.element_size # 2 for K & V
                input_size = B * num_q_per_kv * H_kv * D_h
                output_size = B * num_q_per_kv * H_kv * D_h
                data_transfer_size = KV_cache_size + input_size + output_size # bytes
                arithmetic_intensity = num_flops / data_transfer_size # FLOPS/byte
            prefill_time += operation_time
            prefill_flops += num_flops
            result_dict["prefill"]["breakdown"][op_name] = {#operation_time # ms
                "time": operation_time,
                "flops": num_flops,
                "throughput": throughput,
                "arithmetic_intensity": arithmetic_intensity,
            }

        # decode
        decode_operator = model_operator["decode"]
        decode_time = 0.0
        decode_flops = 0.0
        for op_name, op_data in decode_operator.items():
            if op_data["type"] == "gemm":
                B, M, N, K = op_data["params"]["B"], op_data["params"]["M"], op_data["params"]["N"], op_data["params"]["K"]
                operation_time = self.evaluate_gemm(B, M, N, K)
                num_flops = 2 * M * N * K
                throughput = num_flops / operation_time * 1e-9 # TFLOPS/s
            elif op_data["type"] == "qK_softmax_sV_fused":
                B, H_kv, num_q_per_kv, Lin, D_h = op_data["params"]["B"], op_data["params"]["H_kv"], op_data["params"]["num_q_per_kv"], op_data["params"]["Lin"], op_data["params"]["D_h"]
                operation_time = self.evaluate_qK_softmax_sV_fused(B, H_kv, num_q_per_kv, Lin, D_h)
                num_flops = 2 * (B * H_kv) * num_q_per_kv * Lin * D_h
                throughput = num_flops / operation_time * 1e-9 # TFLOPS/s
                arithmetic_intensity = num_flops / data_transfer_size # FLOPS/byte
                
                KV_cache_size = 2 * B * H_kv * Lin * D_h * self.element_size # 2 for K & V
                input_size = B * num_q_per_kv * H_kv * D_h
                output_size = B * num_q_per_kv * H_kv * D_h
                data_transfer_size = KV_cache_size + input_size + output_size # bytes
            decode_time += operation_time
            decode_flops += num_flops
            result_dict["decode"]["breakdown"][op_name] = {#operation_time # ms
                "time": operation_time,
                "flops": num_flops,
                "throughput": throughput,
            }
        
        result_dict["prefill"]["total_time"] = prefill_time
        result_dict["decode"]["total_time"] = decode_time

        return result_dict
    
    def evaluate_model(self, 
        model, 
        batch_size,
        sum_seq_len,
        gen_seq_len,
    ):
        model_operator_graph = model.build_operator_graph(batch_size, sum_seq_len)
        time_dict_per_layer = self.evaluate_single_layer(model_operator_graph)

        num_layers = model.num_layers
        prefill_time = num_layers * time_dict_per_layer["prefill"]["total_time"] # ms
        decode_time = num_layers * time_dict_per_layer["decode"]["total_time"] * gen_seq_len # ms


        decoding_throughput = batch_size * gen_seq_len / decode_time * 1e3 # tokens/s



        performace = {
            "TTFT_time": prefill_time,
            "decoding_time": decode_time,
            "latency_unit": "ms",
            "decoding_throughput": decoding_throughput,
            "throughput_unit": "tokens/s",
            "time_dict_per_layer": time_dict_per_layer
        }

        return performace


        

