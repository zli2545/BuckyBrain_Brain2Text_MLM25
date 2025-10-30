#!/bin/bash
# Quick Start Script for BuckyBrain Neural Decoder

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}==============================================================${NC}"
echo -e "${GREEN}BuckyBrain Neural Decoder - Quick Start${NC}"
echo -e "${GREEN}==============================================================${NC}"
echo ""

# Check if we're in the right directory
if [ ! -f "train.py" ]; then
    echo -e "${RED}Error: train.py not found. Are you in the project root?${NC}"
    exit 1
fi

# Function to check dependencies
check_dependencies() {
    echo -e "${YELLOW}Checking dependencies...${NC}"
    
    python -c "import torch" 2>/dev/null
    if [ $? -ne 0 ]; then
        echo -e "${RED}Error: PyTorch not installed${NC}"
        echo "Install with: pip install torch torchaudio"
        exit 1
    fi
    
    python -c "import h5py" 2>/dev/null
    if [ $? -ne 0 ]; then
        echo -e "${YELLOW}Warning: h5py not installed${NC}"
        echo "Installing requirements..."
        pip install -r requirements.txt
    fi
    
    echo -e "${GREEN}✓ Dependencies OK${NC}"
    echo ""
}

# Function to show menu
show_menu() {
    echo -e "${GREEN}What would you like to do?${NC}"
    echo ""
    echo "1) Quick debug test (5 minutes, verify setup)"
    echo "2) Small-scale training (30 minutes, test convergence)"
    echo "3) Full training (3-4 hours, reproduce paper results)"
    echo "4) Evaluate existing checkpoint"
    echo "5) Check GPU status"
    echo "6) View training logs"
    echo "7) Install dependencies"
    echo "8) Exit"
    echo ""
}

# Function for debug test
debug_test() {
    echo -e "${GREEN}Running debug test...${NC}"
    echo "This will train for 500 batches (~5 minutes)"
    echo ""
    read -p "GPU to use (default: 0): " gpu
    gpu=${gpu:-0}
    
    python train.py --debug --gpu $gpu
    
    echo ""
    echo -e "${GREEN}Debug test complete!${NC}"
    echo "Check outputs_debug/training.log for details"
    echo ""
}

# Function for small training
small_training() {
    echo -e "${GREEN}Running small-scale training...${NC}"
    echo "This will train for 10,000 batches (~30 minutes)"
    echo ""
    read -p "GPU to use (default: 0): " gpu
    gpu=${gpu:-0}
    
    # Create temporary config
    cat > /tmp/small_config.yaml << EOF
# Include everything from original config, just override batch count
dataset:
  dataset_dir: /mnt/dv/wid/projects3/Rogers-nsf-ind-diff/zihan/brain2text/nejm-brain-to-text/data/hdf5_data_final
  sessions_train: null
  sessions_val: null
  auto_discover: true
  exclude_sessions: []

model:
  n_units: 768
  n_layers: 5
  rnn_dropout: 0.4
  input_dropout: 0.2
  patch_size: 14
  patch_stride: 4

training:
  batch_size: 64
  days_per_batch: 4
  n_train_batches: 10000
  val_every: 1000

optimizer:
  lr: 0.005
  lr_min: 0.0001
  lr_day: 0.005
  weight_decay: 0.001
  weight_decay_day: 0.0
  grad_clip: 10.0

augmentation:
  white_noise_std: 1.0
  constant_offset_std: 0.2
  random_walk_std: 0.0
  static_gain_std: 0.0
  random_cut: 3
  smooth_data: true
  smooth_kernel_std: 2

system:
  output_dir: ./outputs_small
  amp: true
  use_compile: false
  seed: 42
  gpu: 0
EOF
    
    python train.py --config /tmp/small_config.yaml --gpu $gpu --output_dir outputs_small
    
    echo ""
    echo -e "${GREEN}Small training complete!${NC}"
    echo "Check outputs_small/training.log for details"
    echo ""
}

# Function for full training
full_training() {
    echo -e "${GREEN}Running full training...${NC}"
    echo "This will train for 120,000 batches (~3-4 hours)"
    echo ""
    read -p "GPU to use (default: 0): " gpu
    gpu=${gpu:-0}
    
    read -p "Run in background with nohup? (y/n): " use_nohup
    
    if [ "$use_nohup" = "y" ]; then
        nohup python train.py --gpu $gpu > train_full.log 2>&1 &
        PID=$!
        echo ""
        echo -e "${GREEN}Training started in background (PID: $PID)${NC}"
        echo "Monitor with: tail -f train_full.log"
        echo "Or: tail -f outputs/training.log"
        echo ""
    else
        python train.py --gpu $gpu
    fi
}

# Function to evaluate
evaluate() {
    echo -e "${GREEN}Evaluate checkpoint${NC}"
    echo ""
    
    # Find checkpoints
    checkpoints=$(find . -name "best.pt" -o -name "*.pt" | head -5)
    
    if [ -z "$checkpoints" ]; then
        echo -e "${RED}No checkpoints found${NC}"
        echo "Train a model first"
        return
    fi
    
    echo "Available checkpoints:"
    i=1
    for ckpt in $checkpoints; do
        echo "$i) $ckpt"
        i=$((i+1))
    done
    echo ""
    
    read -p "Checkpoint path (or number): " ckpt_input
    
    # If number, convert to path
    if [[ "$ckpt_input" =~ ^[0-9]+$ ]]; then
        ckpt=$(echo "$checkpoints" | sed -n "${ckpt_input}p")
    else
        ckpt=$ckpt_input
    fi
    
    read -p "Dataset split (train/val/test, default: val): " split
    split=${split:-val}
    
    read -p "GPU to use (default: 0): " gpu
    gpu=${gpu:-0}
    
    python eval_phonemes.py \
        --ckpt "$ckpt" \
        --dataset_dir /mnt/dv/wid/projects3/Rogers-nsf-ind-diff/zihan/brain2text/nejm-brain-to-text/data/hdf5_data_final \
        --split $split \
        --gpu $gpu
}

# Function to check GPU
check_gpu() {
    echo -e "${GREEN}GPU Status:${NC}"
    echo ""
    nvidia-smi
    echo ""
}

# Function to view logs
view_logs() {
    echo -e "${GREEN}Recent training logs:${NC}"
    echo ""
    
    # Find log files
    logs=$(find . -name "training.log" -o -name "train*.log" | head -5)
    
    if [ -z "$logs" ]; then
        echo -e "${RED}No log files found${NC}"
        return
    fi
    
    echo "Available logs:"
    i=1
    for log in $logs; do
        echo "$i) $log"
        i=$((i+1))
    done
    echo ""
    
    read -p "Which log to view? (number or path): " log_input
    
    # If number, convert to path
    if [[ "$log_input" =~ ^[0-9]+$ ]]; then
        log=$(echo "$logs" | sed -n "${log_input}p")
    else
        log=$log_input
    fi
    
    echo ""
    echo -e "${YELLOW}Showing last 50 lines of $log${NC}"
    echo -e "${YELLOW}Press 'q' to exit${NC}"
    echo ""
    
    tail -n 50 "$log" | less
}

# Main script
check_dependencies

while true; do
    show_menu
    read -p "Enter choice [1-8]: " choice
    echo ""
    
    case $choice in
        1)
            debug_test
            ;;
        2)
            small_training
            ;;
        3)
            full_training
            ;;
        4)
            evaluate
            ;;
        5)
            check_gpu
            ;;
        6)
            view_logs
            ;;
        7)
            echo "Installing dependencies..."
            pip install -r requirements.txt
            echo -e "${GREEN}✓ Installation complete${NC}"
            echo ""
            ;;
        8)
            echo -e "${GREEN}Goodbye!${NC}"
            exit 0
            ;;
        *)
            echo -e "${RED}Invalid choice. Please enter 1-8${NC}"
            echo ""
            ;;
    esac
    
    read -p "Press Enter to continue..."
    clear
done

