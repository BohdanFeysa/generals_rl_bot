#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <vector>
#include <algorithm>

namespace nb = nanobind;

class BatchedGeneralsEnv {
private:
    int num_envs;
    int h, w;
    // Flat memory layout: [num_envs, 10 channels, h, w]
    // Channel 9 (10th channel) is reserved for the Doomsday clock (Deathtouch)
    std::vector<float> batched_state;
    std::vector<int> current_turns;

public:
    BatchedGeneralsEnv(int num_envs, int h, int w) : num_envs(num_envs), h(h), w(w) {
        batched_state.resize(num_envs * 10 * h * w, 0.0f);
        current_turns.resize(num_envs, 0);
    }

    void reset() {
        std::fill(batched_state.begin(), batched_state.end(), 0.0f);
        std::fill(current_turns.begin(), current_turns.end(), 0);
    }

    void step(const std::vector<int>& actions_cell, const std::vector<int>& actions_dir) {
        #pragma omp parallel for
        for (int i = 0; i < num_envs; ++i) {
            current_turns[i]++;
            
            // 1. Process game logic based on actions_cell[i] and actions_dir[i]
            // ... (Internal grid updates)
            
            // 2. Update the 10th Channel (Deathtouch Clock)
            float time_val = std::min(1.0f, (float)current_turns[i] / 800.0f);
            int channel_offset = i * 10 * h * w + (9 * h * w); 
            std::fill(batched_state.begin() + channel_offset, 
                      batched_state.begin() + channel_offset + (h * w), time_val);
        }
    }

    // Pass the raw memory directly to Python/JAX (Zero-Copy DLPack)
    nb::ndarray<nb::numpy, float> get_batched_observation() {
        size_t shape[4] = { (size_t)num_envs, 10, (size_t)h, (size_t)w };
        return nb::ndarray<nb::numpy, float>(batched_state.data(), 4, shape);
    }
};

NB_MODULE(engine, m) {
    nb::class_<BatchedGeneralsEnv>(m, "BatchedGeneralsEnv")
        .def(nb::init<int, int, int>())
        .def("reset", &BatchedGeneralsEnv::reset)
        .def("step", &BatchedGeneralsEnv::step)
        .def("get_batched_observation", &BatchedGeneralsEnv::get_batched_observation, nb::rv_policy::reference);
}
