from pathlib import Path
import os

def check_and_create_folders():
    # Use the home directory or a user-specific directory to avoid permission issues
    #base_dir = Path.home() / "ml_taxonomy_mapping" / "python_model"
    
    # If you want to use a directory within the current directory instead:
    base_dir = Path(__file__).parent 

    # Ensure the base directory exists first
    base_dir.mkdir(parents=True, exist_ok=True)
    
    folders_to_check = ['mapped_taxonomies', 'base_taxonomy_embeddings', 'base_taxonomy']
    for folder in folders_to_check:
        # Create directories inside the base directory
        folder_path = base_dir / folder
        folder_path.mkdir(parents=True, exist_ok=True)
        print(f"Checked/Created folder: {folder_path}")

if __name__ == "__main__":
    check_and_create_folders()
