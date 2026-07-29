#include <iostream>
#include <vector>
#include <fstream>
#include <cmath>
#include <algorithm>

const int MAP_H = 20, MAP_W = 20;

// Hardcoded Model Weights memory
struct Weights {
    std::vector<float> conv0_w, conv0_b;
    // ... omitting other layers for brevity
};

Weights load_weights(const std::string& filepath) {
    Weights w;
    std::ifstream file(filepath, std::ios::binary);
    if (!file) { std::cerr << "Missing weights.bin" << std::endl; exit(1); }
    
    while(file.peek() != EOF) {
        uint32_t len, size;
        file.read(reinterpret_cast<char*>(&len), sizeof(uint32_t));
        std::string name(len, ' ');
        file.read(&name[0], len);
        
        file.read(reinterpret_cast<char*>(&size), sizeof(uint32_t));
        std::vector<float> data(size);
        file.read(reinterpret_cast<char*>(data.data()), size * sizeof(float));
        
        if (name == "Conv_0.kernel") w.conv0_w = std::move(data);
        else if (name == "Conv_0.bias") w.conv0_b = std::move(data);
    }
    return w;
}

// ----------------------------------------------------
// Fast CP-style Heuristics for Deathtouch Defense
// ----------------------------------------------------
struct Position { int r, c; };
const int dr[] = {-1, 1, 0, 0};
const int dc[] = {0, 0, -1, 1};

bool is_enemy(int r, int c, const std::vector<int>& grid) {
    return grid[r * MAP_W + c] == 2; // Assume 2 is enemy ID
}

void apply_chase_defense_override(int& best_cell, int& best_dir, int general_r, int general_c, const std::vector<int>& grid) {
    // Check 4 adjacent tiles around our general for an attacker
    for (int i = 0; i < 4; ++i) {
        int nr = general_r + dr[i], nc = general_c + dc[i];
        if (nr < 0 || nr >= MAP_H || nc < 0 || nc >= MAP_W) continue;

        if (is_enemy(nr, nc, grid)) {
            // ENEMY DETECTED NEXT TO GENERAL!
            // Execute the Chase Override: Find a third tile to hit their source.
            for (int j = 0; j < 4; ++j) {
                int flank_r = nr + dr[j], flank_c = nc + dc[j];
                if (flank_r == general_r && flank_c == general_c) continue; // Don't hit from the general itself yet
                
                // If we own a tile flanking the attacker, use it!
                if (flank_r >= 0 && flank_r < MAP_H && flank_c >= 0 && flank_c < MAP_W && grid[flank_r * MAP_W + flank_c] == 1) {
                    best_cell = flank_r * MAP_W + flank_c;
                    // Determine attack direction (j reversed)
                    if (dr[j] == -1) best_dir = 1; // hit down
                    else if (dr[j] == 1) best_dir = 0; // hit up
                    else if (dc[j] == -1) best_dir = 3; // hit right
                    else best_dir = 2; // hit left
                    return; 
                }
            }
            
            // Fallback: Clash head on from the General itself. Attacker wins ties, but this is the last resort.
            best_cell = general_r * MAP_W + general_c;
            best_dir = i; 
            return;
        }
    }
}

int main() {
    // 1. Initial 10-second grace period load
    Weights w = load_weights("weights.bin");
    std::vector<float> hidden_state(256, 0.0f);
    
    int turn = 0;
    while (true) {
        // 2. Parse stdin from Competition Server
        // Example logic: std::cin >> turn >> map_data...
        std::vector<int> owners_grid(MAP_H * MAP_W, 0); 
        // ... (parse the grid)
        
        // 3. Manual Math Forward Pass (Takes <5ms on CPU)
        // ... run convolution logic ...
        
        int nn_best_cell = 0; // Placeholder for argmax of neural net output
        int nn_best_dir = 0;
        
        // 4. Turn 800+ Deathtouch Check
        if (turn >= 800) {
            int general_r = 10, general_c = 10; // Placeholder: Fetch from actual grid data
            apply_chase_defense_override(nn_best_cell, nn_best_dir, general_r, general_c, owners_grid);
        }
        
        // 5. Output via stdout to the game runner
        std::cout << nn_best_cell << " " << nn_best_dir << std::endl;
    }
    return 0;
}
