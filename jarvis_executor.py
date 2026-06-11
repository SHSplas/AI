import subprocess
import sys
import os
import time

# Define the path to the command queue file
COMMAND_QUEUE_FILE = "command_queue.txt"
POLL_INTERVAL_SECONDS = 2 # How often to check for new commands
COMMAND_TIMEOUT_SECONDS = 120

def execute_powershell_command(command, timeout=COMMAND_TIMEOUT_SECONDS):
    """
    Executes a given command in PowerShell and returns the output.
    Handles potential errors and provides feedback.
    """
    try:
        # Ensure the command is treated as a string for PowerShell
        # Using -Command to execute a single command
        # Using -NoProfile to speed up startup and avoid profile script interference
        # Using -ExecutionPolicy Bypass for flexibility, but be cautious in production
        # Using encoding='utf-8' to handle various characters
        process = subprocess.run(
            ["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-Command", command],
            capture_output=True,
            text=True,
            encoding='utf-8',
            check=True,  # Raise CalledProcessError if the command returns a non-zero exit code
            timeout=timeout,
        )
        return {
            "success": True,
            "output": process.stdout,
            "error": process.stderr # stderr might still contain warnings even if check=True passes
        }
    except FileNotFoundError:
        return {
            "success": False,
            "output": "",
            "error": "PowerShell executable not found. Please ensure PowerShell is installed and in your PATH."
        }
    except subprocess.CalledProcessError as e:
        return {
            "success": False,
            "output": e.stdout,
            "error": e.stderr
        }
    except subprocess.TimeoutExpired as e:
        return {
            "success": False,
            "output": e.stdout or "",
            "error": f"Command timed out after {timeout} seconds."
        }
    except Exception as e:
        return {
            "success": False,
            "output": "",
            "error": str(e)
        }

def process_commands():
    """Reads commands from the queue file and executes them."""
    if not os.path.exists(COMMAND_QUEUE_FILE):
        print(f"Command queue file '{COMMAND_QUEUE_FILE}' not found. Creating it.")
        with open(COMMAND_QUEUE_FILE, "w", encoding='utf-8') as f:
            f.write("# This file will contain commands for jarvis_executor.py to run.\n")
            f.write("# Example: Start-Process notepad.exe\n")
        return

    commands_to_run = []
    try:
        with open(COMMAND_QUEUE_FILE, "r", encoding='utf-8') as f:
            lines = f.readlines()
            # Filter out empty lines and comments
            commands_to_run = [line.strip() for line in lines if line.strip() and not line.strip().startswith("#")]
    except Exception as e:
        print(f"Error reading command queue file: {e}")
        return

    if not commands_to_run:
        # print("No commands found in queue.") # Suppress noisy output if no commands
        return

    print(f"\n--- Processing {len(commands_to_run)} command(s) from queue ---")

    # Clear the command queue file immediately to prevent re-processing
    try:
        with open(COMMAND_QUEUE_FILE, "w", encoding='utf-8') as f:
            f.write("# This file will contain commands for jarvis_executor.py to run.\n")
            f.write("# Example: Start-Process notepad.exe\n")
    except Exception as e:
        print(f"Error clearing command queue file: {e}")
        # Continue processing commands even if clearing failed, but warn the user

    for command in commands_to_run:
        print(f"Executing: {command}")
        result = execute_powershell_command(command)

        if result["success"]:
            print("--- Command executed successfully ---")
            if result["output"]:
                print("Output:\n", result["output"])
            if result["error"]: # Print stderr even on success, as it may contain warnings
                print("Warnings/Messages:\n", result["error"])
        else:
            print("--- Command failed ---")
            print("Error:\n", result["error"])
            if result["output"]:
                print("Partial Output:\n", result["output"])
        print("------------------------------------------")

def main():
    print("Jarvis Executor started. Monitoring command queue...")
    print(f"Looking for commands in '{COMMAND_QUEUE_FILE}' every {POLL_INTERVAL_SECONDS} seconds.")
    print("Press Ctrl+C to stop.")
    print("------------------------------------------")

    # Ensure the command queue file exists on startup
    if not os.path.exists(COMMAND_QUEUE_FILE):
        print(f"Command queue file '{COMMAND_QUEUE_FILE}' not found. Creating it.")
        try:
            with open(COMMAND_QUEUE_FILE, "w", encoding='utf-8') as f:
                f.write("# This file will contain commands for jarvis_executor.py to run.\n")
                f.write("# Example: Start-Process notepad.exe\n")
        except Exception as e:
            print(f"Failed to create command queue file: {e}")
            return # Exit if we can't even create the queue file

    while True:
        try:
            process_commands()
            time.sleep(POLL_INTERVAL_SECONDS)
        except KeyboardInterrupt:
            print("\nExiting executor.")
            break
        except Exception as e:
            print(f"An unexpected error occurred in the executor loop: {e}")
            # Optionally, break here or continue depending on desired resilience
            time.sleep(POLL_INTERVAL_SECONDS * 2) # Wait longer after an unexpected error

if __name__ == "__main__":
    main()
