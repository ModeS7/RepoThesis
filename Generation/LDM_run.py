#!/usr/bin/env python3
"""
Enhanced master script with real-time progress tracking and colored output.
"""

import subprocess
import sys
import os
import time
import re
from datetime import datetime

try:
    from colorama import init, Fore, Back, Style

    init()  # Initialize colorama for Windows
    COLORS_AVAILABLE = True
except ImportError:
    COLORS_AVAILABLE = False


    # Define dummy color constants
    class Fore:
        RED = GREEN = YELLOW = BLUE = MAGENTA = CYAN = WHITE = RESET = ""


    class Style:
        BRIGHT = DIM = RESET_ALL = ""


def log_message(message, color=None):
    """Log message with timestamp and optional color"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    if COLORS_AVAILABLE and color:
        print(f"{color}[{timestamp}] {message}{Style.RESET_ALL}")
    else:
        print(f"[{timestamp}] {message}")


def parse_progress_info(line, script_name):
    """Extract progress information from output lines"""
    line = line.strip()

    # Look for epoch information
    epoch_match = re.search(r'Epoch (\d+)', line)
    if epoch_match:
        epoch_num = epoch_match.group(1)
        return f"📈 Epoch {epoch_num}"

    # Look for progress percentages
    percent_match = re.search(r'(\d+)%', line)
    if percent_match:
        percent = percent_match.group(1)
        return f"⏳ {percent}% complete"

    # Look for loss values
    loss_match = re.search(r'loss[:\s=]+([0-9]+\.?[0-9]*)', line, re.IGNORECASE)
    if loss_match:
        loss_val = loss_match.group(1)
        return f"📉 Loss: {loss_val}"

    # Look for tqdm progress bars
    if '/it' in line or 'it/s' in line:
        return f"⚡ {line}"

    # Look for file operations
    if 'saving' in line.lower() or 'saved' in line.lower():
        return f"💾 {line}"

    if 'loading' in line.lower() or 'loaded' in line.lower():
        return f"📂 {line}"

    return None


def run_script_with_enhanced_progress(script_path, script_name):
    """Run script with enhanced real-time progress tracking"""
    log_message(f"Starting {script_name}...", Fore.CYAN)

    try:
        process = subprocess.Popen(
            [sys.executable, script_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            universal_newlines=True
        )

        output_lines = []
        current_epoch = None
        last_progress_time = time.time()

        while True:
            output = process.stdout.readline()
            if output == '' and process.poll() is not None:
                break

            if output:
                line = output.strip()
                output_lines.append(output)

                # Parse for progress information
                progress_info = parse_progress_info(line, script_name)

                current_time = time.time()

                if progress_info:
                    # Show enhanced progress info
                    if COLORS_AVAILABLE:
                        print(f"{Fore.GREEN}[{script_name}] {progress_info}{Style.RESET_ALL}")
                    else:
                        print(f"[{script_name}] {progress_info}")
                    last_progress_time = current_time
                else:
                    # Show regular output, but limit frequency for very verbose scripts
                    if (current_time - last_progress_time > 2.0 or  # Show at least every 2 seconds
                            any(keyword in line.lower() for keyword in ['error', 'warning', 'exception', 'failed'])):

                        color = Fore.RED if any(
                            keyword in line.lower() for keyword in ['error', 'exception', 'failed']) else None
                        color = Fore.YELLOW if 'warning' in line.lower() else color

                        if color and COLORS_AVAILABLE:
                            print(f"{color}[{script_name}] {line}{Style.RESET_ALL}")
                        else:
                            print(f"[{script_name}] {line}")

                        last_progress_time = current_time

        return_code = process.poll()

        if return_code == 0:
            log_message(f"{script_name} completed successfully! ✅", Fore.GREEN)
            return True
        else:
            log_message(f"ERROR: {script_name} failed with return code {return_code} ❌", Fore.RED)
            return False

    except FileNotFoundError:
        log_message(f"ERROR: Script not found: {script_path} ❌", Fore.RED)
        return False

    except KeyboardInterrupt:
        log_message(f"Terminating {script_name}... ⏹️", Fore.YELLOW)
        if 'process' in locals():
            process.terminate()
            process.wait()
        raise


def check_model_files_exist():
    """Check if training produced the expected model files"""
    expected_files = [
        r"C:\NTNU\RepoThesis\trained_model\ldm_syn_seg_256_Epoch4_of_5",
        r"C:\NTNU\RepoThesis\trained_model\ldm_conditional_syn_bravo_from_seg_256_Epoch4_of_5"
    ]

    all_exist = True
    for file_path in expected_files:
        if not os.path.exists(file_path):
            log_message(f"WARNING: Expected model file not found: {file_path} ⚠️", Fore.YELLOW)
            all_exist = False
        else:
            file_size = os.path.getsize(file_path) / (1024 * 1024)  # Size in MB
            log_message(f"Found model file: {file_path} ({file_size:.1f} MB) ✅", Fore.GREEN)

    return all_exist


def display_system_info():
    """Display system information for debugging"""
    import torch

    log_message("System Information:", Fore.CYAN)
    print(f"  Python version: {sys.version.split()[0]}")
    print(f"  PyTorch version: {torch.__version__}")
    print(f"  CUDA available: {torch.cuda.is_available()}")

    if torch.cuda.is_available():
        print(f"  CUDA device: {torch.cuda.get_device_name()}")
        print(f"  CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    print()


def main():
    """Main execution function"""
    if COLORS_AVAILABLE:
        log_message("🚀 Starting automated training and inference pipeline", Fore.MAGENTA + Style.BRIGHT)
    else:
        log_message("Starting automated training and inference pipeline")

    # Display system info
    try:
        display_system_info()
    except Exception as e:
        log_message(f"Could not display system info: {e}", Fore.YELLOW)

    # Define script paths
    training_script = "Generation/LDM_training.py"
    inference_script = "Generation/LDM_inference.py"

    # Check if scripts exist
    if not os.path.exists(training_script):
        log_message(f"ERROR: Training script not found: {training_script} ❌", Fore.RED)
        return False

    if not os.path.exists(inference_script):
        log_message(f"ERROR: Inference script not found: {inference_script} ❌", Fore.RED)
        return False

    try:
        start_time = time.time()

        # Step 1: Run training
        log_message("=" * 60, Fore.CYAN)
        log_message("🏋️ PHASE 1: TRAINING", Fore.CYAN + Style.BRIGHT)
        log_message("=" * 60, Fore.CYAN)

        training_start = time.time()
        training_success = run_script_with_enhanced_progress(training_script, "Training")

        if not training_success:
            log_message("Training failed. Stopping pipeline. ❌", Fore.RED)
            return False

        training_time = time.time() - training_start
        log_message(f"Training completed in {training_time / 60:.1f} minutes", Fore.GREEN)

        # Step 2: Verify model files
        log_message("Checking if model files were created...", Fore.CYAN)
        if not check_model_files_exist():
            log_message("Some model files missing, but continuing with inference...", Fore.YELLOW)

        # Step 3: Run inference
        log_message("=" * 60, Fore.CYAN)
        log_message("🔮 PHASE 2: INFERENCE", Fore.CYAN + Style.BRIGHT)
        log_message("=" * 60, Fore.CYAN)

        time.sleep(2)  # Brief pause

        inference_start = time.time()
        inference_success = run_script_with_enhanced_progress(inference_script, "Inference")

        if not inference_success:
            log_message("Inference failed. ❌", Fore.RED)
            return False

        inference_time = time.time() - inference_start
        total_time = time.time() - start_time

        # Success summary
        log_message("=" * 60, Fore.GREEN)
        log_message("🎉 PIPELINE COMPLETED SUCCESSFULLY! 🎉", Fore.GREEN + Style.BRIGHT)
        log_message("=" * 60, Fore.GREEN)
        log_message(f"⏱️  Training time: {training_time / 60:.1f} minutes", Fore.GREEN)
        log_message(f"⏱️  Inference time: {inference_time / 60:.1f} minutes", Fore.GREEN)
        log_message(f"⏱️  Total time: {total_time / 60:.1f} minutes", Fore.GREEN)
        log_message("=" * 60, Fore.GREEN)

        return True

    except KeyboardInterrupt:
        log_message("Pipeline interrupted by user ⏹️", Fore.YELLOW)
        return False

    except Exception as e:
        log_message(f"Unexpected error: {str(e)} ❌", Fore.RED)
        return False


if __name__ == "__main__":
    success = main()
    if not success:
        log_message("Pipeline failed. Check the logs above for details. ❌", Fore.RED)
    sys.exit(0 if success else 1)