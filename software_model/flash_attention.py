from utils import size
from typing import List, Tuple
from hardware_model.device import Device
from software_model.operators import Operator
from software_model.utils import Tensor, DataType
from math import ceil, log2, floor
import torch
import time
import statistics
import numpy as np
import pandas as pd
import os
from scalesim.scale_sim import scalesim
import copy
from tqdm import tqdm

class FlashAttention(Operator):
    def __init__(self, data_type: DataType):
        super().__init__(0, 0, 0, 0, data_type)
        self.Q_shape = None
        self.K_shape = None
        self.V_shape = None
        self.output_shape = None
        self.look_up_table = None
        self.dram_look_up_table = None
        self.mem_name = os.getenv("mem_name")

    def __call__(self, Q: Tensor, K: Tensor, V: Tensor) -> Tensor:
        # Q: [b, h, s, d_h], K: [b, h, d_h, s], V: [b, h, s, d_h]
        # Output: [b, h, s, d_h]
        assert self.data_type == Q.data_type
        assert self.data_type == K.data_type
        assert self.data_type == V.data_type
        
        self.Q_shape = Q.shape
        self.K_shape = K.shape
        self.V_shape = V.shape
        
        # Q: [b, h, s, d_h], K: [b, h, d_h, s], V: [b, h, s, d_h]
        b, h, s, d_h = Q.shape
        assert K.shape == [b, h, d_h, s]
        assert V.shape == [b, h, s, d_h]
        
        self.batch_size = b
        self.num_heads = h
        self.seq_len = s
        self.head_dim = d_h
        
        self.output_shape = [b, h, s, d_h]
        output = Tensor(self.output_shape, self.data_type)
        
        # Create computational graph for flash attention
        self.computational_graph = self.ComputationalGraph(
            self.batch_size, self.num_heads, self.seq_len, self.head_dim, self.data_type
        )
        
        # Flash attention has same FLOP count as conventional attention
        # but different memory access pattern
        self.flop_count = 2 * b * h * s * s * d_h  # Q@K^T + A@V
        self.io_count = b * h * s * d_h * 3  # Q, K, V inputs + O output
        
        return output

    @staticmethod
    def find_permutations(n):
        permutations = set()

        for i in range(1, n + 1):
            if n % i == 0:
                for j in range(1, n + 1):
                    if (n // i) % j == 0:
                        k = n // (i * j)
                        permutations.add((i, j, k))

        return list(permutations)

    def compile_and_simulate(self, pcb_module: Device, compile_mode: str = "heuristic-GPU"):
        if compile_mode != "heuristic-GPU":
            raise ValueError("FlashAttention only supports heuristic-GPU compile mode")
            
        # Initialize lookup tables (reuse from Matmul)
        if self.dram_look_up_table is None:
            filepath = f"./dram_sim_model/{self.mem_name}/look_up_table.csv"
            if not os.path.exists(filepath):
                os.makedirs(os.path.dirname(filepath), exist_ok=True)
                df_empty = pd.DataFrame(columns=["M", "N", "word_size", "Size", "Latency"])
                df_empty.to_csv(filepath, index=False)
                print(f"Created new file at {filepath}")
            self.dram_look_up_table = pd.read_csv(filepath, header=0)

        if self.look_up_table is None:
            self.look_up_table = pd.read_csv(
                f"./systolic_array_model/look_up_table_{pcb_module.compute_module.core.systolic_array.array_height}_{pcb_module.compute_module.core.systolic_array.array_width}.csv",
                header=None,
                names=[
                    "M",
                    "N", 
                    "K",
                    "ArrayHeight",
                    "ArrayWidth",
                    "Dataflow",
                    "cycle_count",
                    "util_rate",
                ],
            )
            self.look_up_table.drop_duplicates(
                inplace=True,
                subset=["M", "N", "K", "ArrayHeight", "ArrayWidth", "Dataflow"],
            )
            self.look_up_table.set_index(
                ["M", "N", "K", "ArrayHeight", "ArrayWidth", "Dataflow"],
                inplace=True,
            )
            
        min_cycle_count = 2**63 - 1
        best_mapping = None
        
        # b = self.computational_graph.batch_size
        # h = self.computational_graph.num_heads
        # s = self.computational_graph.seq_len
        d_h = self.computational_graph.head_dim
        
        # Flash attention block sizes
        for block_size_r in tqdm([32, 64, 128, 256, 512, 1024, 2048]):  # row block size (Br)
            for block_size_c in [32, 64, 128, 256, 512, 1024, 2048]:  # column block size (Bc)
                # Check if blocks fit in SRAM
                # Need space for: Qi (Br x d_h), Kj (Bc x d_h), Vj (Bc x d_h), 
                # Sij (Br x Bc), and output accumulator Oi (Br x d_h)
                working_set_size = (
                    block_size_r * d_h +  # Qi
                    block_size_c * d_h +  # Kj  
                    block_size_c * d_h +  # Vj
                    block_size_r * block_size_c +  # Sij
                    block_size_r * d_h    # Oi accumulator
                )
        
                if (
                    working_set_size
                    > pcb_module.compute_module.l2_size
                    // self.data_type.word_size
                ):
                    continue

                for l1_tile_M in [32, 64, 128, 256]:
                    if l1_tile_M > min(block_size_r, d_h):
                        continue
                    l1_tile_N = l1_tile_M  # L1 tiling is square for simplicity
     
                    for l1_tile_K in [32, 64, 128, 256]:
                        if l1_tile_K > d_h:
                            continue
                        
                        # Check L1 memory constraint for flash attention blocks
                        # Need space for: l1_tile_M*l1_tile_K + l1_tile_K*l1_tile_N + l1_tile_M*l1_tile_N
                        if (
                            l1_tile_M * l1_tile_N
                            + l1_tile_N * l1_tile_K
                            + l1_tile_M * l1_tile_K
                            > pcb_module.compute_module.core.SRAM_size
                            // self.data_type.word_size
                            // 2
                        ):
                            continue                
                        # Add L0 tiling factor exploration like in Matmul
                        for (
                            l0_M_tiling_factor,
                            l0_N_tiling_factor,
                            l0_K_tiling_factor,
                        ) in self.find_permutations(
                            pcb_module.compute_module.core.systolic_array_count
                        ):
                            mapping = self.Mapping(
                                block_size_r=block_size_r,
                                block_size_c=block_size_c,
                                l1_tile_M=l1_tile_M,
                                l1_tile_N=l1_tile_N,
                                l1_tile_K=l1_tile_K,
                                l0_M_tiling_factor=l0_M_tiling_factor,
                                l0_N_tiling_factor=l0_N_tiling_factor,
                                l0_K_tiling_factor=l0_K_tiling_factor
                            )
                            
                            cycle_count = self.simulate(
                                self.computational_graph,
                                mapping,
                                pcb_module,
                            )
                            
                            if cycle_count < min_cycle_count:
                                min_cycle_count = cycle_count
                                best_mapping = mapping
        
        self.best_mapping = best_mapping
        self.best_cycle_count = min_cycle_count
        
        self.best_latency = self.best_cycle_count / pcb_module.compute_module.clock_freq
        self.latency = self.best_latency
        
        return self.latency

    def simulate(
        self,
        computational_graph: "FlashAttention.ComputationalGraph",
        mapping: "FlashAttention.Mapping",
        pcb_module: Device,
    ) -> int:
        b = computational_graph.batch_size
        h = computational_graph.num_heads
        s = computational_graph.seq_len
        d_h = computational_graph.head_dim
        
        Br = mapping.block_size_r
        Bc = mapping.block_size_c
        
        # Number of blocks
        Tr = ceil(s / Br)  # number of row blocks
        Tc = ceil(s / Bc)  # number of column blocks
                
        total_cycle_count = 0
        
        # Use L2TileSimulator for GEMM operations within flash attention blocks
        from software_model.matmul import Matmul
        
        # Create temporary mapping for L2TileSimulator using the L0 tiling factors
        temp_mapping = Matmul.Mapping(
            l2_tile_M=Br,
            l2_tile_N=Bc,
            l2_tile_K=d_h,
            is_l2_double_buffering=True,
            l1_tile_M=mapping.l1_tile_M,
            l1_tile_N=mapping.l1_tile_N,
            l1_tile_K=mapping.l1_tile_K,
            l2_loop_order="knm",
            l1_loop_order="knm",
            l0_M_tiling_factor=mapping.l0_M_tiling_factor,
            l0_N_tiling_factor=mapping.l0_N_tiling_factor,
            l0_K_tiling_factor=mapping.l0_K_tiling_factor
        )
        
        # Pre-compute common L2TileSimulator results to avoid repeated computation
        # Most blocks will have the same size (Br x Bc x d_h), only edge blocks differ
        qk_tile_standard = Matmul.L2TileSimulator(
            Br, Bc, d_h, computational_graph.data_type,
            temp_mapping, pcb_module, self.look_up_table, 
            self.dram_look_up_table, self.mem_name
        )
        av_tile_standard = Matmul.L2TileSimulator(
            Br, d_h, Bc, computational_graph.data_type,
            temp_mapping, pcb_module, self.look_up_table,
            self.dram_look_up_table, self.mem_name
        )
        
        # Pre-compute IO cycles for standard block sizes
        qi_load_cycles_standard = self.simulate_flash_attention_io_cycle_count(
            Br, d_h, computational_graph.data_type, pcb_module
        )
        kj_load_cycles_standard = self.simulate_flash_attention_io_cycle_count(
            Bc, d_h, computational_graph.data_type, pcb_module
        )
        oi_store_cycles_standard = qi_load_cycles_standard  # Same as qi_load
        
        # Pre-compute softmax cycles for standard block size
        softmax_cycles_standard = ceil(
            Br * Bc * 4 / pcb_module.compute_module.total_vector_flops_per_cycle
        )
        
        # Pre-compute init cycles for standard block size
        init_cycles_standard = ceil(Br * d_h / pcb_module.compute_module.total_vector_flops_per_cycle)
        
        # For each row block i
        for i in range(Tr):
            actual_Br = min(Br, s - i * Br)
            
            # Use pre-computed values for standard blocks, compute for edge cases
            if actual_Br == Br:
                init_cycles = init_cycles_standard
                qi_load_cycles = qi_load_cycles_standard
                oi_store_cycles = oi_store_cycles_standard
            else:
                init_cycles = ceil(actual_Br * d_h / pcb_module.compute_module.total_vector_flops_per_cycle)
                qi_load_cycles = self.simulate_flash_attention_io_cycle_count(
                    actual_Br, d_h, computational_graph.data_type, pcb_module
                )
                oi_store_cycles = qi_load_cycles
            
            # For each column block j
            inner_loop_cycles = 0
            for j in range(Tc):
                actual_Bc = min(Bc, s - j * Bc)
                
                # Use pre-computed values for standard blocks, compute for edge cases
                if actual_Bc == Bc:
                    # Standard block size - use pre-computed values
                    kj_load_cycles = kj_load_cycles_standard
                    vj_load_cycles = kj_load_cycles_standard
                    softmax_cycles = softmax_cycles_standard
                    
                    if actual_Br == Br:
                        # Both dimensions are standard - use pre-computed GEMM results
                        sij_compute_cycles = qk_tile_standard.compute_cycle_count
                        accumulator_cycles = av_tile_standard.compute_cycle_count
                    else:
                        # Row dimension is edge case, need to compute
                        qk_tile = Matmul.L2TileSimulator(
                            actual_Br, Bc, d_h, computational_graph.data_type,
                            temp_mapping, pcb_module, self.look_up_table, 
                            self.dram_look_up_table, self.mem_name
                        )
                        av_tile = Matmul.L2TileSimulator(
                            actual_Br, d_h, Bc, computational_graph.data_type,
                            temp_mapping, pcb_module, self.look_up_table,
                            self.dram_look_up_table, self.mem_name
                        )
                        sij_compute_cycles = qk_tile.compute_cycle_count
                        accumulator_cycles = av_tile.compute_cycle_count
                        
                        # Update softmax cycles for edge row case
                        softmax_cycles = ceil(
                            actual_Br * Bc * 4 / pcb_module.compute_module.total_vector_flops_per_cycle
                        )
                else:
                    # Column dimension is edge case - need to compute everything
                    kj_load_cycles = self.simulate_flash_attention_io_cycle_count(
                        actual_Bc, d_h, computational_graph.data_type, pcb_module
                    )
                    vj_load_cycles = kj_load_cycles
                    
                    qk_tile = Matmul.L2TileSimulator(
                        actual_Br, actual_Bc, d_h, computational_graph.data_type,
                        temp_mapping, pcb_module, self.look_up_table, 
                        self.dram_look_up_table, self.mem_name
                    )
                    av_tile = Matmul.L2TileSimulator(
                        actual_Br, d_h, actual_Bc, computational_graph.data_type,
                        temp_mapping, pcb_module, self.look_up_table,
                        self.dram_look_up_table, self.mem_name
                    )
                    sij_compute_cycles = qk_tile.compute_cycle_count
                    accumulator_cycles = av_tile.compute_cycle_count
                    
                    softmax_cycles = ceil(
                        actual_Br * actual_Bc * 4 / pcb_module.compute_module.total_vector_flops_per_cycle
                    )
                
                block_cycles = (
                    kj_load_cycles + vj_load_cycles + sij_compute_cycles + 
                    softmax_cycles + accumulator_cycles
                )
                inner_loop_cycles += block_cycles
            
            # Write Oi back to HBM            
            row_block_cycles = qi_load_cycles + inner_loop_cycles + oi_store_cycles + init_cycles
            total_cycle_count += row_block_cycles
        
        # Account for all batch items and attention heads
        # FlashAttention processes each (batch_item, head) combination independently
        total_cycle_count = total_cycle_count * b * h
        
        return total_cycle_count

    def simulate_flash_attention_io_cycle_count(
        self, M: int, N: int, data_type: DataType, pcb_module: Device
    ):
        # Reuse the DRAM simulation logic from Matmul.L2TileSimulator
        from software_model.matmul import Matmul
        
        # Create a temporary L2TileSimulator just for IO simulation
        temp_mapping = Matmul.Mapping(
            l2_tile_M=M, l2_tile_N=N, l2_tile_K=1,
            is_l2_double_buffering=True,
            l1_tile_M=32, l1_tile_N=32, l1_tile_K=1,
            l2_loop_order="knm", l1_loop_order="knm",
            l0_M_tiling_factor=1, l0_N_tiling_factor=1, l0_K_tiling_factor=1
        )
        
        temp_tile = Matmul.L2TileSimulator(
            M, N, 1, data_type, temp_mapping, pcb_module,
            self.look_up_table, self.dram_look_up_table, self.mem_name
        )
        
        # Return the IO cycle count (use M_K_io_cycle_count as representative)
        return temp_tile.M_K_io_cycle_count

    class ComputationalGraph:
        def __init__(self, batch_size: int, num_heads: int, seq_len: int, head_dim: int, data_type: DataType):
            self.batch_size = batch_size
            self.num_heads = num_heads
            self.seq_len = seq_len
            self.head_dim = head_dim
            self.data_type = data_type

        def display(self):
            print("-" * 10 + " FlashAttention Computational Graph " + "-" * 10)
            print(
                f"Batch: {self.batch_size}, Heads: {self.num_heads}, SeqLen: {self.seq_len}, "
                f"HeadDim: {self.head_dim}, word_size(B): {self.data_type.word_size}"
            )

    class Mapping:
        def __init__(
            self,
            block_size_r: int,
            block_size_c: int, 
            l1_tile_M: int,
            l1_tile_N: int,
            l1_tile_K: int,
            l0_M_tiling_factor: int,
            l0_N_tiling_factor: int,
            l0_K_tiling_factor: int
        ):
            self.block_size_r = block_size_r  # Br - row block size
            self.block_size_c = block_size_c  # Bc - column block size
            self.l1_tile_M = l1_tile_M  # for L1 tiling within blocks
            self.l1_tile_N = l1_tile_N
            self.l1_tile_K = l1_tile_K
            self.l0_M_tiling_factor = l0_M_tiling_factor
            self.l0_N_tiling_factor = l0_N_tiling_factor
            self.l0_K_tiling_factor = l0_K_tiling_factor

        def display(self):
            print(f'{"-"*10} FlashAttention Mapping {"-"*10}')
            print(f"block_size_r: {self.block_size_r}, block_size_c: {self.block_size_c}")
            print(f"l1_tile_M: {self.l1_tile_M}, l1_tile_N: {self.l1_tile_N}, l1_tile_K: {self.l1_tile_K}")
            print(f"l0_M_tiling_factor: {self.l0_M_tiling_factor}, l0_N_tiling_factor: {self.l0_N_tiling_factor}, l0_K_tiling_factor: {self.l0_K_tiling_factor}")