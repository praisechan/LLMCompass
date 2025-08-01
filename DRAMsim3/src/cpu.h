#ifndef __CPU_H
#define __CPU_H

#include <fstream>
#include <functional>
#include <random>
#include <string>
#include "memory_system.h"
#include "configuration.h"

namespace dramsim3 {

class CPU {
   public:
    CPU(const std::string& config_file, const std::string& output_dir,
    const std::string& output_file)
        : config_(new Config(config_file, output_dir, output_file)),
          memory_system_(
              *config_, output_dir,
              std::bind(&CPU::ReadCallBack, this, std::placeholders::_1),
              std::bind(&CPU::WriteCallBack, this, std::placeholders::_1),
              output_file),
          clk_(0) {}
    ~CPU() { delete config_; }
    virtual void ClockTick() = 0;
    void ReadCallBack(uint64_t addr) { return; }
    void WriteCallBack(uint64_t addr) { return; }
    void PrintStats() { memory_system_.PrintStats(); }
    Config *config_;

   protected:
    MemorySystem memory_system_;
    uint64_t clk_;
};

class RandomCPU : public CPU {
   public:
    using CPU::CPU;
    void ClockTick() override;

   private:
    uint64_t last_addr_;
    bool last_write_ = false;
    std::mt19937_64 gen;
    bool get_next_ = true;
};

class StreamCPU : public CPU {
   public:
    using CPU::CPU;
    void ClockTick() override;

   private:
    uint64_t addr_a_, addr_b_, addr_c_, offset_ = 0;
    std::mt19937_64 gen;
    bool inserted_a_ = false;
    bool inserted_b_ = false;
    bool inserted_c_ = false;
    const uint64_t array_size_ = 2 << 20;  // elements in array
    const int stride_ = 64;                // stride in bytes
};

class TraceBasedCPU : public CPU {
   public:
    TraceBasedCPU(const std::string& config_file, const std::string& output_dir,
                  const std::string& output_file,
                  const std::string& trace_file);
    ~TraceBasedCPU() { trace_file_.close(); }
    void ClockTick() override;

   private:
    std::ifstream trace_file_;
    Transaction trans_;
    bool get_next_ = true;
    bool trace_file_end = false;
};

}  // namespace dramsim3
#endif
