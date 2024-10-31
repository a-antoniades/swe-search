import os
import shutil

def move_trajectory_file(root_dir):
    for dirpath, dirnames, filenames in os.walk(root_dir):
        if dirpath.endswith('gpt-4o-mini/rews') and 'trajectory.json' in filenames:
            source_path = os.path.join(dirpath, 'trajectory.json')
            # Construct the destination path
            dest_dir = dirpath.replace('gpt-4o-mini/rews', 'eval_rew_fn/gpt-4o-mini')
            dest_path = os.path.join(dest_dir, 'trajectory.json')
            
            # Ensure the destination directory exists
            os.makedirs(dest_dir, exist_ok=True)
            
            # Move the file
            try:
                shutil.move(source_path, dest_path)
                print(f"Moved: {source_path} -> {dest_path}")
            except FileNotFoundError:
                print(f"Error: Source file not found - {source_path}")
            except PermissionError:
                print(f"Error: Permission denied - {source_path}")
            except Exception as e:
                print(f"Error moving {source_path}: {e}")

# Specify the root directory where the search should start
root_directory = '20240810_search_and_code_gpt-4o-2024-08-06'

# Run the function
move_trajectory_file(root_directory)