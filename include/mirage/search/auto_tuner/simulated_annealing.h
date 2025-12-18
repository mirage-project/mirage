#pragma once

#include <chrono>
#include <cmath>
#include <functional>
#include <random>

#include <iostream>

namespace mirage {
namespace search {

struct SimulatedAnnealingConfig {
  double initial_temperature = 1000.0;
  double final_temperature = 0.01;
  double cooling_rate = 0.8;
  size_t max_iterations = 10000;
  size_t iterations_per_temperature = 10;
  bool use_geometric_cooling = true;
  double time_limit_seconds = -1.0;  // -1.0 means no time limit
  
  SimulatedAnnealingConfig() = default;
  
  SimulatedAnnealingConfig(double init_temp, double final_temp, 
                          double rate, size_t max_iter, size_t iter_per_temp)
      : initial_temperature(init_temp),
        final_temperature(final_temp),
        cooling_rate(rate),
        max_iterations(max_iter),
        iterations_per_temperature(iter_per_temp) {}
};

template <typename StateType, typename EnergyType = double>
class SimulatedAnnealing {
public:
  using State = StateType;
  using Energy = EnergyType;
  
  using InitialStateFunc = std::function<State()>;
  using NeighborFunc = std::function<State(State const &)>;
  using EnergyFunc = std::function<Energy(State const &)>;
  using AcceptanceFunc = std::function<bool(Energy, Energy, double)>;
  
  SimulatedAnnealing(SimulatedAnnealingConfig const &config,
                     InitialStateFunc initial_state_func,
                     NeighborFunc neighbor_func,
                     EnergyFunc energy_func,
                     unsigned int seed = std::random_device{}())
      : config_(config),
        initial_state_func_(initial_state_func),
        neighbor_func_(neighbor_func),
        energy_func_(energy_func),
        acceptance_func_([this](Energy current_energy, Energy neighbor_energy, double temperature) {
          return metropolis_acceptance(current_energy, neighbor_energy, temperature);
        }),
        rng_(seed),
        uniform_dist_(0.0, 1.0),
        temperature_(config.initial_temperature) {}
  
  void set_acceptance_func(AcceptanceFunc acceptance_func) {
    acceptance_func_ = acceptance_func;
  }
  
  State optimize() {
    State current_state = initial_state_func_();
    Energy current_energy = energy_func_(current_state);
    
    State best_state = current_state;
    Energy best_energy = current_energy;
    
    temperature_ = config_.initial_temperature;
    size_t iteration = 0;
    
    auto start_time = std::chrono::steady_clock::now();
    bool has_time_limit = config_.time_limit_seconds > 0.0;
    
    auto time_limit_exceeded = [&]() -> bool {
      if (!has_time_limit) {
        return false;
      }
      auto current_time = std::chrono::steady_clock::now();
      auto elapsed = std::chrono::duration_cast<std::chrono::duration<double>>(
          current_time - start_time).count();
      return elapsed >= config_.time_limit_seconds;
    };
    
    while (temperature_ > config_.final_temperature && 
           iteration < config_.max_iterations) {
      
      if (time_limit_exceeded()) {
        break;
      }

      std::cerr << "temperature: " << temperature_ << std::endl;
      
      for (size_t i = 0; i < config_.iterations_per_temperature; ++i) {
        if (iteration >= config_.max_iterations) {
          break;
        }
        
        if (time_limit_exceeded()) {
          break;
        }
        
        State neighbor_state = neighbor_func_(current_state);
        Energy neighbor_energy = energy_func_(neighbor_state);

        if (neighbor_energy < current_energy) {
          std::cerr << "neighbor_energy: " << neighbor_energy << std::endl;
        }
        
        bool accept = acceptance_func_(current_energy, neighbor_energy, temperature_);
        
        if (accept) {
          current_state = neighbor_state;
          current_energy = neighbor_energy;
          
          if (neighbor_energy < best_energy) {
            best_state = neighbor_state;
            best_energy = neighbor_energy;
          }
        }
        
        ++iteration;
      }
      
      if (config_.use_geometric_cooling) {
        temperature_ *= config_.cooling_rate;
      } else {
        double progress = static_cast<double>(iteration) / static_cast<double>(config_.max_iterations);
        temperature_ = config_.initial_temperature * 
                       std::pow(config_.final_temperature / config_.initial_temperature, progress);
      }
    }
    
    return best_state;
  }
  
  double get_temperature() const { return temperature_; }
  
  void reset_temperature() { temperature_ = config_.initial_temperature; }
  
private:
  SimulatedAnnealingConfig config_;
  InitialStateFunc initial_state_func_;
  NeighborFunc neighbor_func_;
  EnergyFunc energy_func_;
  AcceptanceFunc acceptance_func_;
  
  std::mt19937 rng_;
  std::uniform_real_distribution<double> uniform_dist_;
  double temperature_;
  
  bool metropolis_acceptance(Energy current_energy, Energy neighbor_energy, 
                             double temperature) {
    Energy delta = neighbor_energy - current_energy;
    
    if (delta < 0) {
      return true;
    }
    
    double probability = std::exp(-delta / temperature);
    return uniform_dist_(rng_) < probability;
  }
};

} // namespace search
} // namespace mirage

