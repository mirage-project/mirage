from PIL import Image
import re
import os

def combine(h):
    # Load the two images
    image1 = Image.open(f'/home/kitao/projects/mirage/dataset/demo/vis/original/{h}.png')
    image2 = Image.open(f'/home/kitao/projects/mirage/dataset/demo/vis/optimized/{h}.png')

    # Make sure the heights match (optional: you can resize if needed)
    if image1.height != image2.height:
        # Resize image2 to match image1's height (keeping aspect ratio)
        new_height = image1.height
        new_width = int(image2.width * (new_height / image2.height))
        image2 = image2.resize((new_width, new_height))

    # Create a new blank image with combined width
    combined_width = image1.width + image2.width
    combined_image = Image.new('RGBA', (combined_width, image1.height))

    # Paste images side by side
    combined_image.paste(image1, (0, 0))
    combined_image.paste(image2, (image1.width, 0))

    # Save the combined image
    combined_image.save(f'/home/kitao/projects/mirage/dataset/demo/vis/combined/{h}.png')