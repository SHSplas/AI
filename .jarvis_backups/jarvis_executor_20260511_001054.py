import subprocess
import sys
import os

def execute_powershell_command(command):
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
            check=True  # Raise CalledProcessError if the command returns a non-zero exit code
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
    except Exception as e:
        return {
            "success": False,
            "output": "",
            "error": str(e)
        }

def main():
    print("Jarvis Executor started. Waiting for commands...")
    print("You can now ask Jarvis to execute commands.")
    print("Example: Ask Jarvis to 'run notepad.exe'")
    print("------------------------------------------")

    # In a real-world scenario, this script would need a way to
    # receive commands from Jarvis. For this example, we'll simulate
    # a loop where Jarvis can send commands (e.g., via a shared file,
    # or by you copying/pasting commands I provide into this terminal).

    # For now, we'll use a simple input loop to demonstrate.
    # Jarvis will provide the command string to you, and you'll paste it here.
    while True:
        try:
            # This is where Jarvis would ideally send commands directly.
            # Since direct communication isn't built-in, we'll simulate it
            # by asking the user to input the command that Jarvis would provide.
            command_to_run = input("Enter command for Jarvis (or 'exit' to quit): ")
            if command_to_run.lower() == 'exit':
                break
            if not command_to_run:
                continue

            print(f"\nExecuting: {command_to_run}")
            result = execute_powershell_command(command_to_run)

            if result["success"]:
                print("\n--- Command executed successfully ---")
                if result["output"]:
                    print("Output:\n", result["output"])
                if result["error"]: # Print stderr even on success, as it may contain warnings
                    print("Warnings/Messages:\n", result["error"])
            else:
                print("\n--- Command failed ---")
                print("Error:\n", result["error"])
                if result["output"]:
                    print("Partial Output:\n", result["output"])

            print("------------------------------------------")

        except KeyboardInterrupt:
            print("\nExiting executor.")
            break
        except Exception as e:
            print(f"An unexpected error occurred in the executor loop: {e}")
            break

if __name__ == "__main__":
    main()