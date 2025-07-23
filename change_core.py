from software_model.transformer import (
    TransformerBlockInitComputationTP,
    TransformerBlockAutoRegressionTP,
)
from software_model.utils import data_type_dict, Tensor
from design_space_exploration.dse import template_to_system, read_architecture_template
from multiprocessing import Process, Lock
from cost_model.cost_model import calc_compute_chiplet_area_mm2, calc_io_die_area_mm2
import time, os


def run(overall_config):
    run_name, device_type, model_type, input_seq_length, batch_size, output_seq_length = overall_config
    print(f"Current config is {model_type}, {input_seq_length}, {batch_size}, {output_seq_length}\n")

    model_config = {
        "Llama-2-7B": {"d_model":4096, "n_heads":32, "intermediate_dim":11008, "n_layers":32},
        "OPT-30B": {"d_model":7168, "n_heads":56, "intermediate_dim":28672, "n_layers":48},
    }

    # Get the configuration based on defaulting to default_config
    model_config = model_config.get(model_type)

    # Extract the values from the config
    d_model = model_config["d_model"]
    n_heads = model_config["n_heads"]
    intermediate_dim = model_config["intermediate_dim"]
    n_layers = model_config["n_layers"]

    ## DSE template to system
    if device_type == "A100":
      arch_specs = read_architecture_template("configs/In-Flash/GA100_40GB.json")
      compute_mode = "no_QK_AV"
    elif device_type == "H100":
      arch_specs = read_architecture_template("configs/In-Flash/GH100_80GB.json")
      compute_mode = "no_QK_AV"
    elif device_type == "A100_In_Flash":
      arch_specs = read_architecture_template("configs/In-Flash/GA100_40GB.json")
      compute_mode = "no_QK_AV_softmax"
    elif device_type == "H100_In_Flash":
      arch_specs = read_architecture_template("configs/In-Flash/GH100_80GB.json")
      compute_mode = "no_QK_AV_softmax"
      
    device_count = arch_specs["device_count"]
    mem_name = arch_specs["device"]["memory_protocol"]
    os.environ['mem_name'] = mem_name    

    model_init = TransformerBlockInitComputationTP(
        d_model=d_model,
        n_heads=n_heads,
        intermediate_dim=intermediate_dim,
        device_count=device_count,
        data_type=data_type_dict["fp16"],
    )
    model_auto_regression = TransformerBlockAutoRegressionTP(
        d_model=d_model,
        n_heads=n_heads,
        intermediate_dim=intermediate_dim,
        device_count=device_count,
        data_type=data_type_dict["fp16"],
        compute_mode=compute_mode
    )
    _ = model_init(
        Tensor([batch_size, input_seq_length, model_init.d_model], data_type_dict["fp16"])
    )
    _ = model_auto_regression(
        Tensor([batch_size, 1, model_init.d_model], data_type_dict["fp16"]),
        input_seq_length + output_seq_length,
    )


    def test_core(name, lock):
        compute_area_mm2 = calc_compute_chiplet_area_mm2(arch_specs)
        io_area_mm2 = calc_io_die_area_mm2(arch_specs)
        print(f"{name}, {compute_area_mm2}, {io_area_mm2}, {compute_area_mm2+io_area_mm2}")
        # exit()
        system = template_to_system(arch_specs)

        dir = f"./{run_name}/{device_type}/{model_type}"
        os.makedirs(dir, exist_ok=True)
        
        auto_regression_latency_simulated = model_auto_regression.compile_and_simulate(
            system, "heuristic-GPU"
        )
        init_latency_simulated = model_init.compile_and_simulate(system, "heuristic-GPU")
        print(f"{name}, {init_latency_simulated}, {auto_regression_latency_simulated}")
        with lock:
            with open(f"{dir}/core_size_results_init.csv", "a") as f:
                f.write(
                    f"{name}, {compute_area_mm2+io_area_mm2}, {init_latency_simulated}, {init_latency_simulated*n_layers}, {model_init.simluate_log}\n"
                )
            with open(f"{dir}/core_size_results_ar.csv", "a") as f:
                f.write(
                    f"{name}, {compute_area_mm2+io_area_mm2}, {auto_regression_latency_simulated}, {auto_regression_latency_simulated*n_layers}, {model_auto_regression.simluate_log}\n"
                )

    lock = Lock()
    simul_name = f"Batch{batch_size}_Input{input_seq_length}_Output{output_seq_length}"
    test_core(simul_name, lock)

    # lock = Lock()
    # configs = [
    #     ("A", 128, 4, 8, 8, 192),
    #     ("B", 128, 4, 16, 32, 192),
    #     ("C", 128, 1, 32, 128, 192),
    #     ("D", 32, 1, 64, 512, 768),
    #     ("E", 8, 1, 128, 2048, 3072),
    # ]

    # processes = [Process(target=test_core_size, args=(i, lock)) for i in configs]

    # try:
    #     for p in processes:
    #         p.start()

    #     while any(p.is_alive() for p in processes):
    #         time.sleep(1)
    # except KeyboardInterrupt:
    #     print("Terminating processes...")
    #     for p in processes:
    #         p.terminate()
    #         p.join()

overall_configs = [
    # case_name, device_type, model_type, input_seq_length, batch_size, output_seq_length
    #####Llama 7B#####
    ("SOM_Llama-2-7B", "A100", "Llama-2-7B", 256, 1, 1), #input 256 / batch 1 / output 1
    ("SOM_Llama-2-7B", "A100", "Llama-2-7B", 256, 2, 1), #input 256 / batch 1 / output 1
    ("SOM_Llama-2-7B", "A100", "Llama-2-7B", 256, 4, 1), #input 256 / batch 4 / output 1
    ("SOM_Llama-2-7B", "A100", "Llama-2-7B", 256, 8, 1), #input 256 / batch 4 / output 1
    ("SOM_Llama-2-7B", "A100", "Llama-2-7B", 256, 16, 1), #input 256 / batch 16 / output 1
    ("SOM_Llama-2-7B", "A100", "Llama-2-7B", 256, 32, 1), #input 256 / batch 16 / output 1
    ("SOM_Llama-2-7B", "A100", "Llama-2-7B", 256, 64, 1), #input 256 / batch 64 / output 1
    ]

for i in overall_configs:
    run(i)

print("All processes have finished.")
