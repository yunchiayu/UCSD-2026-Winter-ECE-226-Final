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
        data_transfer_size = weight_size + input_size # bytes
        memory_time = data_transfer_size / peak_bandwidth * 1e-9 # ms

        # GEMM time
        total_flops = 2 * M * N * K
        gemm_time = total_flops / peak_compute * 1e-9 # ms

        total_time = max(memory_time, gemm_time)

        return total_time

    def evaluate_single_layer(self, model_operator: dict):
        time_dict = {
            "prefill": {
                "total_time": 0.0,
                "breakdown": {
                    "unit": "us",
                }
            },
            "decode": {
                "total_time": 0.0,
                "breakdown": {
                    "unit": "us",
                }
            }
        }
        # prefill
        prefill_operator = model_operator["prefill"]
        prefill_time = 0.0
        for op_name, op_data in prefill_operator.items():
            B, M, N, K = op_data["B"], op_data["M"], op_data["N"], op_data["K"]
            operation_time = self.evaluate_gemm(B, M, N, K)
            prefill_time += operation_time
            time_dict["prefill"]["breakdown"][op_name] = operation_time * 1e3 # us

        # decode
        decode_operator = model_operator["decode"]
        decode_time = 0.0
        for op_name, op_data in decode_operator.items():
            B, M, N, K = op_data["B"], op_data["M"], op_data["N"], op_data["K"]
            operation_time = self.evaluate_gemm(B, M, N, K)
            decode_time += operation_time
            time_dict["decode"]["breakdown"][op_name] = operation_time
        
        time_dict["prefill"]["total_time"] = prefill_time
        time_dict["decode"]["total_time"] = decode_time

        return time_dict
    
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


        

