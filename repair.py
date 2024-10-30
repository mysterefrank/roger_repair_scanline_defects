import replicate
import numpy as np
import cv2
from PIL import Image
import os
from io import BytesIO
from pathlib import Path
import time

def detect_scanlines(image, sensitivity=1.15, min_line_length=30, max_line_width=3):
    """
    Detect scan lines in image and create mask
    """
    # Convert image to numpy array
    img_array = np.array(image)
    
    # Convert to grayscale for detection
    if len(img_array.shape) == 3:
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    else:
        gray = img_array.copy()
    
    # Apply slight contrast enhancement for better line detection
    gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)
    
    # Create empty mask
    mask = np.zeros_like(gray)
    
    # Detect scan lines by comparing each row with rows above and below
    height, width = gray.shape
    for y in range(max_line_width, height-max_line_width):
        # Look at surrounding rows within max_line_width
        surrounding_rows = []
        for offset in range(1, max_line_width + 1):
            surrounding_rows.extend([gray[y-offset], gray[y+offset]])
        surrounding_avg = np.mean(surrounding_rows, axis=0)
        
        # Detect bright lines
        brighter_pixels = gray[y] > (surrounding_avg * sensitivity)
        
        if np.any(brighter_pixels):
            # Convert to uint8 for connectedComponents
            brighter_pixels_uint8 = brighter_pixels.astype(np.uint8)
            
            # Label connected components
            num_labels, labels = cv2.connectedComponents(brighter_pixels_uint8)
            
            # Check each component's length
            for label in range(1, num_labels):
                component_positions = np.where(labels == label)[0]
                
                if len(component_positions) >= min_line_length:
                    # Set those positions to 255 in the mask
                    mask[y, component_positions] = 255
    
    # Dilate mask slightly
    kernel = np.ones((2,2), np.uint8)  # Reduced kernel size
    mask = cv2.dilate(mask, kernel, iterations=1)
    
    return Image.fromarray(mask)

def interpolate_scanlines(image_array, mask_array):
    """
    Fill scanlines using interpolation from pixels above and below
    """
    result = image_array.copy()
    height, width = mask_array.shape[:2]
    
    # For each row
    for y in range(1, height-1):
        # Find scanline pixels in this row
        scanline_pixels = mask_array[y] > 0
        if np.any(scanline_pixels):
            # For each channel if color image
            if len(image_array.shape) == 3:
                for c in range(image_array.shape[2]):
                    # Get values above and below
                    above_values = image_array[y-1, scanline_pixels, c]
                    below_values = image_array[y+1, scanline_pixels, c]
                    # Average them
                    result[y, scanline_pixels, c] = (above_values + below_values) // 2
            else:
                # Grayscale image
                above_values = image_array[y-1, scanline_pixels]
                below_values = image_array[y+1, scanline_pixels]
                result[y, scanline_pixels] = (above_values + below_values) // 2
    
    return result

def detect_bad_repairs(original, repaired, mask, threshold=30):
    """
    Detect areas where the AI repair looks bad
    """
    bad_repair_mask = np.zeros_like(mask)
    
    # Convert to grayscale if needed
    if len(original.shape) == 3:
        original_gray = cv2.cvtColor(original, cv2.COLOR_RGB2GRAY)
        repaired_gray = cv2.cvtColor(repaired, cv2.COLOR_RGB2GRAY)
    else:
        original_gray = original
        repaired_gray = repaired
    
    height, width = mask.shape[:2]
    
    for y in range(1, height-1):
        scanline_pixels = mask[y] > 0
        if np.any(scanline_pixels):
            # Get surrounding pixel values
            above_pixels = original_gray[y-1][scanline_pixels]
            below_pixels = original_gray[y+1][scanline_pixels]
            surrounding_avg = (above_pixels + below_pixels) / 2
            
            # Check if repaired pixels deviate too much from expected values
            repaired_pixels = repaired_gray[y][scanline_pixels]
            difference = np.abs(repaired_pixels - surrounding_avg)
            bad_pixels = difference > threshold
            
            # Mark bad repairs in the mask
            bad_repair_mask[y][scanline_pixels] = np.where(bad_pixels, 255, 0)
    
    return bad_repair_mask

def repair_scanlines(image_path, output_path, sensitivity=1.15, min_line_length=30):
    """
    Repair scan lines using both AI and interpolation
    """
    print(f"Processing {image_path}...")
    
    try:
        print("Loading image...")
        original_image = Image.open(image_path)
        original_array = np.array(original_image)
        
        print("Detecting scanlines...")
        mask = detect_scanlines(original_image, sensitivity, min_line_length)
        mask_array = np.array(mask)
        
        # Save mask temporarily
        mask_path = f"debug_mask_{Path(image_path).stem}.png"
        mask.save(mask_path)
        
        print("Running AI inpainting...")
        output = replicate.run(
            "stability-ai/stable-diffusion-inpainting:95b7223104132402a9ae91cc677285bc5eb997834bd2349fa486f53910fd68b3",
            input={
                "mask": open(mask_path, "rb"),
                "image": open(image_path, "rb"),
                "width": 768,
                "height": 576,
                "prompt": "grainy nigerian horror movie made for tv",
                "scheduler": "DDIM",
                "num_outputs": 1,
                "guidance_scale": 7.49,
                "negative_prompt": "white block of pixels, white, discontinuities",
                "num_inference_steps": 500
            }
        )
        
        # Process the AI-inpainted image
        print("Processing AI result...")
        inpainted_data = output[0].read()
        inpainted_image = Image.open(BytesIO(inpainted_data))
        inpainted_array = np.array(inpainted_image)
        
        # Add this new code here to handle RGBA to RGB conversion
        if original_array.shape[-1] == 4:  # If original has alpha channel
            original_array = original_array[..., :3]  # Keep only RGB channels
        if inpainted_array.shape[-1] == 4:  # If inpainted has alpha channel
            inpainted_array = inpainted_array[..., :3]  # Keep only RGB channels
            
        # Resize if needed
        if original_array.shape != inpainted_array.shape:
            inpainted_array = cv2.resize(inpainted_array, (original_array.shape[1], original_array.shape[0]))
        
        # Take darker pixels from AI result
        ai_result = np.minimum(original_array, inpainted_array)
        
        # Detect bad repairs
        print("Checking repair quality...")
        bad_repair_mask = detect_bad_repairs(original_array, ai_result, mask_array)
        
        # Save debug visualization of bad repairs
        debug_path = f"debug_bad_repairs_{Path(image_path).stem}.png"
        Image.fromarray(bad_repair_mask).save(debug_path)
        
        # Use interpolation for bad repairs
        print("Fixing bad repairs with interpolation...")
        final_result = ai_result.copy()
        interpolated = interpolate_scanlines(original_array, bad_repair_mask)
        
        # Where repairs were bad, use interpolated values instead
        if len(final_result.shape) == 3:
            for c in range(final_result.shape[2]):
                final_result[:,:,c] = np.where(bad_repair_mask > 0, 
                                             interpolated[:,:,c], 
                                             final_result[:,:,c])
        else:
            final_result = np.where(bad_repair_mask > 0, 
                                  interpolated, 
                                  final_result)
        
        # Save the result
        print(f"Saving result to {output_path}")
        final_image = Image.fromarray(final_result)
        final_image.save(output_path)
        
        # Clean up temporary files
        os.remove(mask_path)
        os.remove(debug_path)
        
        print(f"Successfully processed {image_path}")
        return True
        
    except Exception as e:
        print(f"Error processing {image_path}: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def process_directory(input_dir, output_dir, sleep_time=1):
    """
    Process all images in a directory
    """
    # Create output directory if it doesn't exist
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Get all image files
    image_extensions = ('.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG')
    image_files = [f for f in os.listdir(input_dir) if f.endswith(image_extensions)]
    total_files = len(image_files)
    
    print(f"Found {total_files} images to process")
    
    # Process each image
    successful = 0
    failed = 0
    
    for idx, filename in enumerate(sorted(image_files), 1):
        input_path = os.path.join(input_dir, filename)
        output_path = os.path.join(output_dir, f"repaired_{filename}")
        
        print(f"\nProcessing image {idx}/{total_files}: {filename}")
        
        if repair_scanlines(input_path, output_path):
            successful += 1
        else:
            failed += 1
        
        if idx < total_files:
            print(f"Sleeping for {sleep_time} seconds...")
            time.sleep(sleep_time)
    
    print(f"\nProcessing complete!")
    print(f"Successfully processed: {successful}")
    print(f"Failed: {failed}")
    print(f"Total: {total_files}")

if __name__ == "__main__":
    
    process_directory(
        input_dir="frames_dir",
        output_dir="output_dir",
        sleep_time=1
    )