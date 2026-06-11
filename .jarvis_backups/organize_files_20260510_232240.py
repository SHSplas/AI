import os
import shutil
import csv
import glob

def create_and_move(src_path, dest_dir):
    """Creates destination directory if it doesn't exist and moves the file."""
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)
    if os.path.exists(src_path):
        try:
            shutil.move(src_path, os.path.join(dest_dir, os.path.basename(src_path)))
            # print(f"Moved: {src_path} -> {dest_dir}") # Suppress print for cleaner output
        except Exception as e:
            print(f"Could not move {src_path} to {dest_dir}: {e}")
    # else:
        # print(f"Skipping: {src_path} (not found)") # Suppress print for cleaner output

def organize_project_files():
    print("Starting file organization...")

    # Define directories
    dirs = {
        "scripts": "scripts",
        "data": "data",
        "data_legacy": "data/legacy",
        "models": "models",
        "metrics": "metrics",
        "metrics_temp_archive": "metrics/temp_archive",
        "logs": "logs",
        "eval_reports": "eval_reports",
        "temp": "temp" # New temporary folder
    }
    for d in dirs.values():
        os.makedirs(d, exist_ok=True)

    # List of files to move
    files_to_organize = {
        # Scripts
        "GPT_model.py": dirs["scripts"],
        "build_mixed_refine_data.py": dirs["scripts"],
        "chat.py": dirs["scripts"],
        "eval_chat_prompts.py": dirs["scripts"],
        "fetch_wikidata_qa.py": dirs["scripts"],
        "input.py": dirs["scripts"],
        "jarvis_pretrained_chat.py": dirs["scripts"],
        "mixer.py": dirs["scripts"],
        "parallel_bpe.py": dirs["scripts"],
        "prepare_data.py": dirs["scripts"],
        "prepare_refine_data.py": dirs["scripts"],
        "train.py": dirs["scripts"],

        # Data
        "bpe_vocab.json": dirs["data"],
        "bpe_vocab_legacy_20260213.json": dirs["data_legacy"],
        "input.txt": dirs["data"], # This is a raw source file
        os.path.join("data", "Easy.txt"): dirs["data"],
        os.path.join("data", "Medium.txt"): dirs["data"],
        os.path.join("data", "Hard.txt"): dirs["data"],
        os.path.join("data", "jarvis_train.txt"): dirs["data"],
        os.path.join("data", "jarvis_val.txt"): dirs["data"],
        os.path.join("data", "jarvis_mix_train.txt"): dirs["data"],
        os.path.join("data", "jarvis_refine_train.txt"): dirs["data"],
        os.path.join("data", "jarvis_refine_val.txt"): dirs["data"],
        os.path.join("data", "jarvis_data_report.json"): dirs["data"],
        os.path.join("data", "jarvis_mix_report.json"): dirs["data"],
        os.path.join("data", "jarvis_refine_report.json"): dirs["data"],
        os.path.join("data", "jarvis_eval_prompts.txt"): dirs["data"],
        os.path.join("data", "web_wikidata_qa.txt"): dirs["data"],
        "mixed_train.txt": dirs["data"], # Output from mixer.py
    }

    # Models
    model_files = glob.glob("*.pth")
    for model_file in model_files:
        files_to_organize[model_file] = dirs["models"]

    # Metrics
    metrics_files = glob.glob("*.csv")
    for metrics_file in metrics_files:
        if metrics_file.startswith("temp_"):
            files_to_organize[metrics_file] = dirs["metrics_temp_archive"]
        else:
            files_to_organize[metrics_file] = dirs["metrics"]

    # Eval Reports
    eval_json_files = glob.glob("cycle_eval_*.json") + glob.glob("manual_eval_*.json") + glob.glob("rebuild_eval_*.json")
    for ef in eval_json_files:
        files_to_organize[ef] = dirs["eval_reports"]
        # Also move corresponding .txt files if they exist
        if os.path.exists(ef.replace(".json", ".txt")):
            files_to_organize[ef.replace(".json", ".txt")] = dirs["eval_reports"]

    # Logs (initial pass for general logs and completion times)
    log_files = glob.glob("train_*.log") + glob.glob("train_*.err") + glob.glob("samples.txt")
    completion_files = glob.glob("*_completion_time.txt")
    for lf in log_files + completion_files:
        files_to_organize[lf] = dirs["logs"]

    # Perform initial moves
    for src, dest in files_to_organize.items():
        if os.path.exists(src):
            create_and_move(src, dest)
        elif os.path.exists(os.path.join(os.getcwd(), src)): # Check if it's in data/ or other subdirs
             create_and_move(os.path.join(os.getcwd(), src), dest)

    # Consolidate logs and temporary metrics into the new 'temp' folder
    print("\nConsolidating temporary and log files into 'temp/'...")
    for file_name in os.listdir(dirs["logs"]):
        create_and_move(os.path.join(dirs["logs"], file_name), dirs["temp"])
    if os.path.exists(dirs["logs"]) and not os.listdir(dirs["logs"]):
        shutil.rmtree(dirs["logs"])
        print(f"Removed empty directory: {dirs['logs']}")

    for file_name in os.listdir(dirs["metrics_temp_archive"]):
        create_and_move(os.path.join(dirs["metrics_temp_archive"], file_name), dirs["temp"])
    if os.path.exists(dirs["metrics_temp_archive"]) and not os.listdir(dirs["metrics_temp_archive"]):
        shutil.rmtree(dirs["metrics_temp_archive"])
        print(f"Removed empty directory: {dirs['metrics_temp_archive']}")

    print("File organization complete!")

def find_best_model():
    metrics_dir = "metrics"
    model_metrics = {}

    for metrics_file in glob.glob(os.path.join(metrics_dir, "*.csv")):
        try:
            with open(metrics_file, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    val_loss_str = row.get("val_loss")
                    if val_loss_str:
                        try:
                            val_loss = float(val_loss_str)
                            step = int(row.get("step", 0))
                            model_name = os.path.basename(metrics_file).replace("_metrics.csv", "_best.pth")
                            
                            # Use the model name to track best val_loss
                            if model_name not in model_metrics or val_loss < model_metrics[model_name]["val_loss"]:
                                model_metrics[model_name] = {
                                    "val_loss": val_loss,
                                    "step": step,
                                    "metrics_file": metrics_file
                                }
                        except ValueError:
                            continue # Skip rows with non-numeric val_loss

        except Exception as e:
            print(f"Error reading {metrics_file}: {e}")

    if not model_metrics:
        print("No model metrics found to determine the best model.")
        return

    best_overall_model = None
    lowest_val_loss = float('inf')

    for model_name, data in model_metrics.items():
        if data["val_loss"] < lowest_val_loss:
            lowest_val_loss = data["val_loss"]
            best_overall_model = model_name

    if best_overall_model:
        print("\n--- Best Model Based on Validation Loss ---")
        print(f"Model: {best_overall_model}")
        print(f"Validation Loss: {model_metrics[best_overall_model]['val_loss']:.4f}")
        print(f"Recorded at Step: {model_metrics[best_overall_model]['step']}")
        print(f"From Metrics File: {model_metrics[best_overall_model]['metrics_file']}")
    else:
        print("\nCould not determine a best model based on validation loss.")

    print("\nYour project's designated important model: models/cpu_gpt_jarvis_big_v1_best.pth (never deleted)")


if __name__ == "__main__":
    organize_project_files()
    find_best_model()