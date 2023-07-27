WEIGHT_DIR="weights"

if [ ! -d "$DIRECTORY" ]; then
    mkdir weights
fi
cd weights

# Download 15M tinystories model from Andrej Karpathy
wget https://huggingface.co/karpathy/tinyllamas/resolve/main/stories15M.bin
