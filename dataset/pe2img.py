import os
import numpy as np
from PIL import Image
import math
from tqdm import tqdm

def convert_pe_to_rgb(input_folder, output_folder, target_size=320):
    """
    Convert PE files to RGB images with error handling and progress tracking
    """
    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)
    
    # Get list of valid files
    valid_extensions = ('.exe', '.dll')
    files = [f for f in os.listdir(input_folder) if f.lower().endswith(valid_extensions)]
    
    # Process each PE file with progress bar
    for filename in tqdm(files, desc="Converting files"):
        file_path = os.path.join(input_folder, filename)
        output_path = os.path.join(output_folder, f"{filename}.png")
        
        try:
            # Check if file exists and is not empty
            if not os.path.exists(file_path) or os.path.getsize(file_path) == 0:
                print(f"Skipping {filename}: File is empty or doesn't exist")
                continue
                
            # Read binary data in chunks to handle large files
            with open(file_path, 'rb') as f:
                binary_data = f.read()
            
            # Convert binary to array of integers
            binary_array = np.frombuffer(binary_data, dtype=np.uint8)
            
            # Calculate required size for target RGB image
            required_size = target_size * target_size * 3
            
            # Pad or truncate the array to match required size
            if len(binary_array) < required_size:
                padding = required_size - len(binary_array)
                binary_array = np.pad(binary_array, (0, padding), 'constant')
            else:
                binary_array = binary_array[:required_size]
            
            # Reshape to target_size x target_size x 3
            rgb_array = binary_array.reshape((target_size, target_size, 3))
            
            # Create and save image
            image = Image.fromarray(rgb_array)
            image.save(output_path, optimize=True)
            
        except Exception as e:
            print(f"Error converting {filename}: {str(e)}")
            continue

    print(f"\nConversion completed. Images saved to: {output_folder}")
    print(f"Total files processed: {len(files)}")

def verify_images(output_folder, target_size=320):
    """
    Verify that all generated images are valid and have correct dimensions
    """
    print("\nVerifying generated images...")
    invalid_images = []
    
    for filename in os.listdir(output_folder):
        if filename.endswith('.png'):
            image_path = os.path.join(output_folder, filename)
            try:
                with Image.open(image_path) as img:
                    if img.size != (target_size, target_size):
                        invalid_images.append(filename)
            except Exception:
                invalid_images.append(filename)
    
    if invalid_images:
        print(f"Found {len(invalid_images)} invalid images:")
        for img in invalid_images:
            print(f"- {img}")
    else:
        print("All images are valid and have correct dimensions")

if __name__ == "__main__":
    # Example usage
    inp=str(input("Enter location to images: "))
    out=str(input("Enter location to save images: "))
    input_folder = inp
    output_folder = out
    
    # Convert files
    convert_pe_to_rgb(input_folder, output_folder)
    
    # Verify generated images
    verify_images(output_folder)
