#!/bin/bash
# Exit on any failure
set -e

echo "Starting compilation of Contest Bot..."

# Compile the bot with maximum CPU optimizations, statically linking everything.
# We enforce C++20 and strict warnings to ensure CP code is safe.
g++ -O3 -std=c++20 -march=native -Wall -Wextra contest_bot.cpp -o bot_executable

# Ensure the weights file is present in the final directory
if [ ! -f "weights.bin" ]; then
    echo "ERROR: weights.bin missing! Make sure to bundle it in your zip."
    exit 1
fi

echo "Build complete. Ready for battle."
