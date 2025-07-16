#!/usr/bin/env python3

import argparse
import os
import random


class Generator():
    """
    Format agnostic address stream generator
    """

    def __init__(self, stream_type, interarrival, ratio, gb, shift_bits, num_ch=1, ch_pos=-1, num_bank=1, bank_pos=(-1,-1), mem_type = None, ro_pos=-1):
        # convert to 0 ~ 1 for easier random generation
        self._interval = interarrival
        self._ratio = ratio / (ratio + 1.0)

        self._gen = None

        self._range = gb * (2 ** 30)
        self._last_clk = 0
        self._last_rd_addr = random.randrange(self._range)
        self._last_wr_addr = random.randrange(self._range)
        if stream_type == 'random':
            self._gen = self._rand_gen
            # self._gen = self._rand_read
        elif stream_type == 'stream':
            self._gen = self._stream_gen
        elif stream_type == 'readonly_stream':
            self._last_rd_addr = 0
            self._gen = self._readonly_stream_gen
        elif stream_type == 'readonly_mult_ch_stream': # channel interleaving
            self._gen = self._readonly_mult_ch_stream_gen
            # self._last_rd_addr_arr = [0]*num_ch
            # for i in range(num_ch):
            #   self._last_rd_addr_arr[i] += (2**ch_pos) * i #ch_pos: index of which channel bit starts from.
            assert ch_pos != -1, 'ch_pos should be assigned for multiple channel'
            self._last_rd_addr_arr = [0]*num_bank
            self.bank_counter = [0]*num_bank
            for i in range(num_bank):
                  self._last_rd_addr_arr[i] += (2**bank_pos[0]) * i #bank_pos: index of which channel bit starts from.
            assert bank_pos[0] != -1, 'bank_pos should be assigned for bank interleaving'
        elif stream_type == 'readonly_bank_interleave_stream':
            self._gen = self._readonly_bank_interleave_stream_coarse
            self._last_rd_addr_arr = [0]*num_bank
            self.bank_counter = [0]*num_bank
            for i in range(num_bank):
                  self._last_rd_addr_arr[i] += (2**bank_pos[0]) * i #bank_pos: index of which channel bit starts from.
            assert bank_pos[0] != -1, 'bank_pos should be assigned for bank interleaving'
        else:
            self._gen = self._mix_gen

        self.shift_bits = shift_bits
        self.num_ch = num_ch
        self.last_ch = -1
        self.ch_pos = ch_pos
        self.ro_pos = ro_pos
        
        self.num_bank = num_bank
        self.last_bank = 0
        self.bank_pos = bank_pos
        self.channel_0_addr = None
        self.mem_type = mem_type
    
    def _get_op(self):
        if random.random() > self._ratio:
            return 'w'
        else:
            return 'r'

    def _rand_gen(self):
        addr = random.randrange(self._range)
        op = self._get_op()
        return (op, addr)

    def _rand_read(self):
        # addr = random.randrange(self._range)
        addr = random.randrange((2**30) * self.num_bank)
        op = 'r'
        return (op, addr)

    def _stream_gen(self):
        op = self._get_op()
        if op == 'r':
            self._last_rd_addr += self.shift_bits
            return (op, self._last_rd_addr)
        else:
            self._last_wr_addr += self.shift_bits
            return (op, self._last_wr_addr)

    def _readonly_stream_gen(self):
        self._last_rd_addr += self.shift_bits
        return ('r', self._last_rd_addr)
    
    def _readonly_bank_interleave_stream_coarse(self):       
        if self.bank_counter[self.last_bank] == 0:
            self.bank_counter[self.last_bank] +=1
        else:
            self.bank_counter[self.last_bank] +=1
            self._last_rd_addr_arr[self.last_bank] += self.shift_bits
            
            if self.mem_type=="LPDDR4":
              #if address exceeds bank field, switch to another row
              if self._last_rd_addr_arr[self.last_bank] % 2**self.bank_pos[0] == 0:
                  self._last_rd_addr_arr[self.last_bank] -= 2**self.bank_pos[0]
                  self._last_rd_addr_arr[self.last_bank] += 2**self.ro_pos
            elif self.mem_type=="HBM2":
              #if address exceeds channel field, switch to another row
              if self._last_rd_addr_arr[self.last_bank] % 2**self.ch_pos == 0:
                  self._last_rd_addr_arr[self.last_bank] -= 2**self.ch_pos
                  self._last_rd_addr_arr[self.last_bank] += 2**self.ro_pos
    
        final_addr = self._last_rd_addr_arr[self.last_bank]

        print(f"bank:{self.last_bank}, bank_counter: {self.bank_counter[self.last_bank]}, addr: {final_addr}")

        # switch bank for 8 reads
        if self.bank_counter[self.last_bank] % 8 == 0:
            if self.last_bank==self.num_bank-1:
                self.last_bank = 0
            else: 
                self.last_bank += 1
        
        return ('r', final_addr)

    # def _readonly_bank_interleave_stream_coarse(self):
    #     # switch bank for 32 reads
    #     if self.bank_counter[self.last_bank] % 32 == 0 and self.bank_counter[self.last_bank] != 0:
    #         if self.last_bank==self.num_bank-1:
    #             self.last_bank = 0
    #         else: 
    #             self.last_bank += 1
        
    #     if self.bank_counter[self.last_bank] != 0:
    #         self._last_rd_addr_arr[self.last_bank] += self.shift_bits

    #     #if address exceeds bank field, add higher bit above bank field(rank or something)
    #     if self._last_rd_addr_arr[self.last_bank] % 2**self.bank_pos[0] == 0 and self.bank_counter[self.last_bank] != 0:
    #         self._last_rd_addr_arr[self.last_bank] -= 2**self.bank_pos[0]
    #         self._last_rd_addr_arr[self.last_bank] += 2**self.bank_pos[1]

    #     self.bank_counter[self.last_bank] +=1
        
    #     return ('r', self._last_rd_addr_arr[self.last_bank])
        
    def _readonly_mult_ch_stream_gen(self):
        if self.last_ch == -1 or self.last_ch==self.num_ch-1:
            # increase address for channel 0
            self.last_ch = 0
            _, self.channel_0_addr = self._readonly_bank_interleave_stream_coarse()
            return ('r', self.channel_0_addr)
        else:
            # increment channel index by 1 and reuse the address of channel 0
            self.last_ch += 1
            return ('r', self.channel_0_addr + (2**self.ch_pos) * self.last_ch)

    # def _readonly_mult_ch_stream_gen_old(self):
    #     if self.last_ch == -1 or self.last_ch==self.num_ch-1:
    #         # increase address for channel 0
    #         self.last_ch = 0
            
    #         if self.bank_counter[self.last_bank] != 0:
    #           self._last_rd_addr_arr[self.last_bank] += self.shift_bits
              
    #           #if address exceeds bank field, add higher bit above bank field(rank or something)
    #           if self._last_rd_addr_arr[self.last_bank] % 2**self.bank_pos[0] == 0:
    #               self._last_rd_addr_arr[self.last_bank] -= 2**self.bank_pos[0]
    #               self._last_rd_addr_arr[self.last_bank] += 2**self.bank_pos[1]
      
    #         self.bank_counter[self.last_bank] +=1

    #         #if address exceeds channel field, roll back to initial value        
    #         if self._last_rd_addr_arr[self.last_bank] >= 2**self.ch_pos: #if address exceeds channel field
    #           self._last_rd_addr_arr[self.last_bank] -= 2**self.ch_pos
    #           self._last_rd_addr_arr[self.last_bank] += 2**self.ch_pos[1]

    #         self.channel_0_addr = self._last_rd_addr_arr[self.last_bank]

    #         print(f"bank:{self.last_bank}, bank_counter: {self.bank_counter[self.last_bank]}, addr: {self.channel_0_addr}")

    #         # switch bank for 8 reads
    #         if self.bank_counter[self.last_bank] % 8 == 0:
    #             if self.last_bank==self.num_bank-1:
    #                 self.last_bank = 0
    #             else: 
    #                 self.last_bank += 1
            
    #         return ('r', self.channel_0_addr)
    #     else:
    #         # increment channel index by 1 and reuse the address of channel 0
    #         self.last_ch += 1
    #         return ('r', self.channel_0_addr + (2**self.ch_pos) * self.last_ch)

    def _mix_gen(self):
        if random.random() > 0.5:
            return self._rand_gen()
        else:
            return self._stream_gen()

    def gen(self):
        op, addr = self._gen()
        self._last_clk += self._interval
        return (op, addr, self._last_clk)


def get_string(op, addr, clk, trace_format, interarrival):
    op_map = {
        'r': {
            'dramsim2': 'READ',
            'dramsim3': 'READ',
            'ramulator': 'R',
            'usimm': 'R',
            'drsim': 'READ'
        },
        'w': {
            'dramsim2': 'WRITE',
            'dramsim3': 'WRITE',
            'ramulator': 'W',
            'usimm': 'W',
            'drsim': 'WRITE'
        }
    }
    actual_op = op_map[op][trace_format]
    if 'dramsim' in trace_format:
        return '{} {} {}\n'.format(hex(addr), actual_op, clk)
    elif 'ramulator' == trace_format:
        return '{} {}\n'.format(hex(addr), actual_op)
    elif 'usimm' == trace_format:
        # USIMM assumes a 3.2GHz CPU by default, we hard code it here...
        # also use clk for pc for convinience
        if actual_op == 'R':
            return '{} {} {} 0x0\n'.format(interarrival, actual_op, hex(addr))
        else:
            return '{} {} {}\n'.format(interarrival, actual_op, hex(addr))
    elif 'drsim' == trace_format:
        return '{} {} {} 64B\n'.format(hex(addr), actual_op, clk)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Trace Generator for Various DRAM Simulators",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-s', '--stream-type', default='random',
                        help='Address stream type, (r)andom, (s)tream, (m)ix, readonly_stream,\
                        writeonly_stream, readonly_bank_interleave_stream, readonly_mult_ch_stream,\
                        readonly_interleaving_bank_sweep')
    parser.add_argument('-i', '--interarrival',
                        help='Inter-arrival time in cycles',
                        type=int, default=0)
    parser.add_argument('-f', '--format', default='dramsim3',
                        help='Trace format, dramsim2, dramsim3,'
                        'ramulator, usimm, drsim, or all')
    parser.add_argument("-o", "--output-dir",
                        help="output directory", default=".")
    parser.add_argument('-r', '--ratio', type=float, default=2,
                        help='Read to write(1) ratio')
    parser.add_argument('-n', '--num-reqs', type=int, default=100,
                        help='Total number of requests.')
    parser.add_argument('-g', '--gb', type=int, default=4,
                        help='GBs of address space')
    parser.add_argument('-b', '--shift-bits', type=int, default=64,
                        help='shift bits, how much to increment address in stream mode')
    parser.add_argument('--num-bank', type=int, default=1,
                        help='number of used banks. how many banks are used or interleaved')
    parser.add_argument('--num-ch', type=int, default=1,
                        help='number of used channels. how many banks are used or interleaved')
    parser.add_argument('--mem-type', type=str, default=None,
                        help='memory to use. DDR4, LPDDR4, HBM2')

    args = parser.parse_args()

    if not os.path.exists(args.output_dir):
        try:
            os.mkdir(args.output_dir)
        except (OSError, ValueError) as e:
            print('Cannot use output path:' + args.output_dir)
            print(e)
            exit(1)
    print("Output directory: ", args.output_dir)

    stream_types = {'r': 'random', 'random': 'random',
                    's': 'stream', 'stream': 'stream',
                    'm': 'mix', 'mix': 'mix',
                    'readonly_stream':'readonly_stream',
                    'writeonly_stream':'writeonly_stream',
                    'c':'readonly_mult_ch_stream','readonly_mult_ch_stream':'readonly_mult_ch_stream',
                    'b':'readonly_bank_interleave_stream', '_readonly_bank_interleave_stream':'readonly_bank_interleave_stream',
                    # 'a': 'readonly_interleaving_bank_sweep', 'readonly_interleaving_bank_sweep':'readonly_interleaving_bank_sweep'
                    }
    stream_type = stream_types.get(args.stream_type, 'random')
    print("Address stream type: ", stream_type)

    formats = ['dramsim2', 'dramsim3', 'ramulator', 'usimm', 'drsim']
    if args.format != 'all':
        formats = [args.format]
    print("Trace format(s):", formats)

    files = {}
    for f in formats:
        file_name = '{}_{}_i{}_n{}_rw{}.trace'.format(
            f, stream_type, args.interarrival, args.num_reqs, int(args.ratio))
        if f == 'dramsim2':
            file_name = 'mase_' + file_name
        print("Write to file: ", file_name)
        files[f] = os.path.join(args.output_dir, file_name)

    # open files
    for f, name in files.items():
        fp = open(name, 'w')
        files[f] = fp

    if args.mem_type == "DDR4":
      g = Generator(stream_type, args.interarrival, args.ratio, args.gb, args.shift_bits, num_ch=2, ch_pos=17, num_bank=args.num_bank, bank_pos=(13, 16), mem_type="DDR4")#DDR4
    elif args.mem_type == "LPDDR4":
      g = Generator(stream_type, args.interarrival, args.ratio, args.gb, shift_bits=128, num_ch=args.num_ch, ch_pos=17, num_bank=args.num_bank, bank_pos=(13, 16), ro_pos=19, mem_type="LPDDR4")#LPDDR4_4ch
    elif args.mem_type == "LPDDR4_mobile":
      g = Generator(stream_type, args.interarrival, args.ratio, args.gb, shift_bits=32, num_ch=args.num_ch, ch_pos=15, num_bank=args.num_bank, bank_pos=(11, 14), ro_pos=17, mem_type="LPDDR4")#LPDDR4_4ch_mobile
    elif args.mem_type == "HBM2":
      # HBM' mapping is rorabgbachco; channel comes earlier than bank
      g = Generator(stream_type, args.interarrival, args.ratio, args.gb, shift_bits=64, num_ch=args.num_ch, ch_pos=11, num_bank= 4, bank_pos=(18, 20), ro_pos = 19, mem_type="HBM2")#HBM2_32ch
    else:
      raise ValueError("Invalid memory name.")
  
    for i in range(args.num_reqs):
        op, addr, clk = g.gen()
        for f in formats:
            line = get_string(op, addr, clk, f, args.interarrival)
            files[f].write(line)