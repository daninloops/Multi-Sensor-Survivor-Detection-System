import os
import cv2
import hashlib
import shutil
from src.paths import RAW_DIR, PROCESSED_DIR, LOGS_DIR

def get_image_hash(image_path):
    with open(image_path, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()

def clean_dataset():
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    report_file = os.path.join(LOGS_DIR, "dataset_report.txt")
    
    total_images = 0
    corrupted_count = 0
    duplicate_count = 0
    processed_count = 0
    
    hashes = set()
    
    # Categories: human_present, no_human
    categories = ["human_present", "no_human"]
    
    for category in categories:
        category_raw = os.path.join(RAW_DIR, category)
        category_processed = os.path.join(PROCESSED_DIR, category)
        os.makedirs(category_processed, exist_ok=True)
        
        if not os.path.exists(category_raw):
            print(f"Warning: {category_raw} does not exist. Skipping.")
            continue
            
        for root, dirs, files in os.walk(category_raw):
            for file in files:
                if not file.lower().endswith((".jpg", ".jpeg", ".png", ".webp")):
                    continue
                    
                total_images += 1
                img_path = os.path.join(root, file)
                
                # 1. Check if corrupted
                img = cv2.imread(img_path)
                if img is None:
                    print(f"Removing corrupted image: {file}")
                    os.remove(img_path)
                    corrupted_count += 1
                    continue
                
                # 2. Check for duplicates
                img_hash = get_image_hash(img_path)
                if img_hash in hashes:
                    print(f"Removing duplicate image: {file}")
                    os.remove(img_path)
                    duplicate_count += 1
                    continue
                
                hashes.add(img_hash)
                
                # 3. Resize and Save to Processed
                img_resized = cv2.resize(img, (224, 224))
                dst_path = os.path.join(category_processed, file)
                cv2.imwrite(dst_path, img_resized)
                processed_count += 1

    # Generate Report
    with open(report_file, "w") as f:
        f.write("--- Dataset Cleaning Report ---\n")
        f.write(f"Total Images Analyzed: {total_images}\n")
        f.write(f"Corrupted Images Removed: {corrupted_count}\n")
        f.write(f"Duplicate Images Removed: {duplicate_count}\n")
        f.write(f"Final Processed Dataset Size: {processed_count}\n")
        f.write("-------------------------------\n")

    print(f"\nCleanup Complete. Report saved to {report_file}")

if __name__ == "__main__":
    clean_dataset()
