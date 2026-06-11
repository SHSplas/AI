import os
import random
from datasets import load_dataset

# 1. SETUP FILENAMES
movie_script = 'input.txt'      # Your original overnight training file
output_file = 'mixed_train.txt'

# 2. DOWNLOAD ALPACA DATA (Instruction Following)
print("Downloading Alpaca Dataset from Hugging Face...")
alpaca = load_dataset("tatsu-lab/alpaca", split='train')

def format_entry(entry):
    """Formats Alpaca JSON into User/Assistant text."""
    instruction = entry['instruction']
    context = entry['input']
    response = entry['output']
    
    if context:
        return f"User: {instruction}\nContext: {context}\nAssistant: {response}\n\n"
    else:
        return f"User: {instruction}\nAssistant: {response}\n\n"

# 3. GATHER DATA
mixed_data = []

print("Processing 50,000 chat examples...") # how many examples to mix in
for i in range(50000):
    mixed_data.append(format_entry(alpaca[i]))

# 4. MIX IN MOVIE SCRIPT
if os.path.exists(movie_script):
    print(f"Mixing in your {movie_script}")
    with open(movie_script, 'r', encoding='utf-8') as f:
        movie_text = f.read()
        # We break the movie into blocks to shuffle them with the chat data
        movie_blocks = [block.strip() + "\n\n" for block in movie_text.split('\n\n') if block.strip()]
        mixed_data.extend(movie_blocks)
else:
    print(f" Warning: {movie_script} not found. Proceeding with Chat data only.")

# 5. SHUFFLE & SAVE
print(" Shuffling data for better learning...")
random.shuffle(mixed_data)

with open(output_file, 'w', encoding='utf-8') as f_out:
    f_out.write("".join(mixed_data))

print(f"SUCCESS! Created '{output_file}'.")
