import cv2
import glob
import os
import argparse

# Set up argument parsing
parser = argparse.ArgumentParser(description='Resize all JPEG images in a specified directory.')
parser.add_argument('-path', '--path', type=str, required=True, help='Directory containing JPEG files to resize.')
parser.add_argument('-width', '--width', type=int, required=True, help='New width for the resized images.')
parser.add_argument('-height', '--height', type=int, required=True, help='New height for the resized images.')
args = parser.parse_args()

# Extract the width, height, and path from the arguments
new_width = args.width
new_height = args.height
target_directory = args.path


# Ensure the target directory is an absolute path
target_directory = os.path.abspath(target_directory)

# Create a pattern to match all jpg files in the target directory
pattern = os.path.join(target_directory, '*.jpg')

# Use glob to get all the jpg files in that directory
jpg_files = glob.glob(pattern)

print(f"Resize to {new_width}x{new_height} - fouhd {jpg_files} images")
# Loop through all the files, read and resize them using OpenCV
for jpg_file in jpg_files:
    # Read the image
    img = cv2.imread(jpg_file)
    
    # Check if image is read correctly
    if img is not None:
        # Resize the image
        resized_image = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_AREA)
        
        # Extract the base name without its extension
        base_name = os.path.splitext(os.path.basename(jpg_file))[0]
        resized_file_name = f"{base_name}_{new_width}x{new_height}.jpg"
        resized_file_path = os.path.join(target_directory, resized_file_name)
        cv2.imwrite(resized_file_path, resized_image)
        print(f"Resizing {jpg_file}")
        
    else:
        print(f"Error reading image {jpg_file}")

# Close any open windows
cv2.destroyAllWindows()