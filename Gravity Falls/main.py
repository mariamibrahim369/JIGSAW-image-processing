import os
import cv2
import numpy as np
import json
import shutil


class Config:
    # Input: ONE sample image (change this to your image path)
    IMAGE_PATH = "Gravity Falls/puzzle_8x8/0.jpg"  # Single image file
    
    # Output paths
    OUTPUT_BASE = "outputs_m1"
    OUTPUT_TILES = "outputs_m1/tiles"
    OUTPUT_EDGES = "outputs_m1/edges"
    OUTPUT_CONTOURS = "outputs_m1/contours"
    OUTPUT_VISUALS = "outputs_m1/visualizations"
    CONTOUR_FILE = "outputs_m1/contours.json"
    
    # Processing parameters
    GAUSSIAN_KERNEL = (5, 5)
    MEDIAN_KERNEL = 3
    CLAHE_CLIP = 2.0
    CLAHE_GRID = (8, 8)
    GAMMA = 1.2
    CANNY_LOW = 50
    CANNY_HIGH = 150
    MIN_CONTOUR_AREA = 100
    
    # Puzzle configurations
    PUZZLE_SIZES = {
        "2x2": 2,
        "4x4": 4,
        "8x8": 8
    }

# ===============================
# SETUP
# ===============================
def setup_directories():
    """Create output directories."""
    folders = [
        Config.OUTPUT_TILES,
        Config.OUTPUT_EDGES,
        Config.OUTPUT_CONTOURS,
        Config.OUTPUT_VISUALS
    ]
    
    for folder in folders:
        if os.path.exists(folder):
            shutil.rmtree(folder)
        os.makedirs(folder, exist_ok=True)
    
    print("Output directories created/cleared")

def detect_grid_size(image_path):
    """Detect grid size from filename or folder name. Fallback to 2x2."""
    filename = os.path.basename(image_path).lower()
    for size_key, size_val in Config.PUZZLE_SIZES.items():
        if size_key in filename:
            return size_val
    
    folder_name = os.path.basename(os.path.dirname(image_path)).lower()
    for size_key, size_val in Config.PUZZLE_SIZES.items():
        if size_key in folder_name:
            return size_val
    
    print(f"Warning: Could not detect grid size, defaulting to 2x2")
    return 2

# ===============================
# PHASE 1: PREPROCESSING
# ===============================
def preprocess_image(img, gamma=Config.GAMMA):
    """Preprocessing pipeline applied to the entire image."""
    print("  Applying Gaussian blur...")
    blur = cv2.GaussianBlur(img, Config.GAUSSIAN_KERNEL, 0)
    
    print("  Converting to grayscale...")
    gray = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)
    
    print("  Applying median blur...")
    median = cv2.medianBlur(gray, Config.MEDIAN_KERNEL)
    
    print("  Applying CLAHE enhancement...")
    clahe = cv2.createCLAHE(clipLimit=Config.CLAHE_CLIP, tileGridSize=Config.CLAHE_GRID)
    enhanced = clahe.apply(median)
    
    print("  Applying sharpening filter...")
    sharpening_kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=np.float32)
    sharp = cv2.filter2D(enhanced, -1, sharpening_kernel)
    
    print("  Applying gamma correction...")
    gamma_corrected = np.power(sharp / 255.0, gamma) * 255.0
    gamma_corrected = np.clip(gamma_corrected, 0, 255).astype(np.uint8)
    
    # Back to BGR for consistent handling
    preprocessed = cv2.cvtColor(gamma_corrected, cv2.COLOR_GRAY2BGR)
    
    return preprocessed

# ===============================
# PHASE 3: CONTOUR EXTRACTION
# ===============================
def extract_contours(edges):
    """Exact same simple contour extraction as your original."""
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours

# ===============================
# FEATURE EXTRACTION
# ===============================
def get_contour_features(contour):
    """Extract features needed for future matching."""
    features = {}
    
    features['area'] = float(cv2.contourArea(contour))
    features['perimeter'] = float(cv2.arcLength(contour, True))
    
    x, y, w, h = cv2.boundingRect(contour)
    features['bounding_box'] = [int(x), int(y), int(w), int(h)]
    
    M = cv2.moments(contour)
    if M['m00'] != 0:
        cx = int(M['m10'] / M['m00'])
        cy = int(M['m01'] / M['m00'])
        features['centroid'] = [cx, cy]
    else:
        features['centroid'] = [0, 0]
    
    hu_moments = cv2.HuMoments(M).flatten()
    features['hu_moments'] = [float(h) for h in hu_moments]
    
    epsilon = 0.01 * features['perimeter']
    approx = cv2.approxPolyDP(contour, epsilon, True)
    features['num_vertices'] = len(approx)
    
    return features

# ===============================
# MAIN PROCESSING
# ===============================
def process_image(image_path):
    """Full pipeline: preprocess → divide → detect edges & contours per tile."""
    img = cv2.imread(image_path)
    if img is None:
        print(f"Could not load {image_path}")
        return None
    
    filename = os.path.basename(image_path)
    print(f"\nProcessing: {filename}")
    
    GRID_SIZE = detect_grid_size(image_path)
    print(f"Grid size: {GRID_SIZE}x{GRID_SIZE}")
    
    # PHASE 1: Preprocess entire image
    print("\nPHASE 1: Preprocessing entire image...")
    preprocessed_img = preprocess_image(img)
    base_name = os.path.splitext(filename)[0]
    cv2.imwrite(os.path.join(Config.OUTPUT_VISUALS, f"{base_name}_preprocessed.png"), preprocessed_img)
    print("Full image preprocessed")
    
    # PHASE 2: Divide into tiles
    print("\nPHASE 2: Dividing into tiles...")
    h, w, _ = preprocessed_img.shape
    tile_h = h // GRID_SIZE
    tile_w = w // GRID_SIZE
    
    # For full reconstruction using preprocessed colors
    full_contour_img = np.zeros_like(preprocessed_img)
    
    # Grid visualization on original image
    grid_viz = img.copy()
    for i in range(1, GRID_SIZE):
        cv2.line(grid_viz, (0, i * tile_h), (w, i * tile_h), (0, 0, 255), 2)
        cv2.line(grid_viz, (i * tile_w, 0), (i * tile_w, h), (0, 0, 255), 2)
    
    tile_count = 0
    all_tile_data = {}
    
    # PHASE 3: Process each tile
    print("\nPHASE 3: Edge detection and contour extraction per tile...")
    
    for row in range(GRID_SIZE):
        for col in range(GRID_SIZE):
            y1, y2 = row * tile_h, (row + 1) * tile_h
            x1, x2 = col * tile_w, (col + 1) * tile_w
            
            # Use preprocessed tile for everything visual
            tile = preprocessed_img[y1:y2, x1:x2]
            
            # 1. Save tile
            tile_filename = f"tile_{tile_count}.png"
            cv2.imwrite(os.path.join(Config.OUTPUT_TILES, tile_filename), tile)
            
            # 2. Edge detection
            gray = cv2.cvtColor(tile, cv2.COLOR_BGR2GRAY)
            blur = cv2.GaussianBlur(gray, (5, 5), 0)
            edges = cv2.Canny(blur, Config.CANNY_LOW, Config.CANNY_HIGH)
            
            edge_filename = f"edges_{tile_count}.png"
            cv2.imwrite(os.path.join(Config.OUTPUT_EDGES, edge_filename), edges)
            
            # 3. Find contours
            contours = extract_contours(edges)
            
            # 4. Draw contours on PREPROCESSED tile (this is the key change)
            contour_img = tile.copy()
            cv2.drawContours(contour_img, contours, -1, (0, 255, 0), 3)
            
            contour_filename = f"contour_{tile_count}.png"
            cv2.imwrite(os.path.join(Config.OUTPUT_CONTOURS, contour_filename), contour_img)
            
            # 5. Paste into full reconstruction (also using preprocessed colors)
            full_contour_img[y1:y2, x1:x2] = contour_img
            
            # 6. Collect data for JSON
            contours_data = []
            for cnt in contours:
                if cv2.contourArea(cnt) < Config.MIN_CONTOUR_AREA:
                    continue
                contours_data.append({
                    'points': cnt.tolist(),
                    'features': get_contour_features(cnt)
                })
            
            all_tile_data[tile_count] = {
                "tile_file": tile_filename,
                "edge_file": edge_filename,
                "contour_file": contour_filename,
                "position": [int(x1), int(y1)],
                "grid_position": [row, col],
                "num_contours": len(contours_data),
                "contours": contours_data
            }
            
            tile_count += 1
    
    print(f"All {tile_count} tiles processed")
    
    # Save final visualizations
    cv2.imwrite(os.path.join(Config.OUTPUT_VISUALS, f"{base_name}_grid.png"), grid_viz)
    cv2.imwrite(os.path.join(Config.OUTPUT_VISUALS, f"{base_name}_contours_full.png"), full_contour_img)
    
    total_contours = sum(t['num_contours'] for t in all_tile_data.values())
    print(f"Total valid contours found: {total_contours}")
    
    return all_tile_data

# ===============================
# MAIN
# ===============================
def main():
    print("=" * 60)
    print("MILESTONE 1: JIGSAW PUZZLE PREPROCESSING")
    print("=" * 60)
    
    setup_directories()
    
    if not os.path.isfile(Config.IMAGE_PATH):
        print(f"Image not found or not a file: {Config.IMAGE_PATH}")
        print("\nTip: Update IMAGE_PATH in Config class to point to your puzzle image.")
        return
    
    print(f"Input image: {Config.IMAGE_PATH}")
    print("Processing order: Full preprocess → Divide → Edge/Contour per tile\n")
    
    result = process_image(Config.IMAGE_PATH)
    
    if result is None:
        print("\nProcessing failed!")
        return
    
    # Save JSON
    output_data = {os.path.basename(Config.IMAGE_PATH): result}
    with open(Config.CONTOUR_FILE, 'w') as f:
        json.dump(output_data, f, indent=4)
    
    print("\n" + "=" * 60)
    print("PROCESSING COMPLETE!")
    print("=" * 60)
    print(f"Output folder: {Config.OUTPUT_BASE}")
    print(f"JSON data:     {Config.CONTOUR_FILE}")
    print("\nContents:")
    print("  tiles/          → Preprocessed pieces (with enhancement)")
    print("  edges/          → Canny edges")
    print("  contours/       → Contours drawn on enhanced pieces")
    print("  visualizations/ → Full preprocessed, grid, and contour overlay")
    print("  contours.json   → All contour points + features")
    print("\nTo process another image → just change IMAGE_PATH")
    print("=" * 60)

if __name__ == "__main__":   # Fixed: was _name_ → __name__
    main()