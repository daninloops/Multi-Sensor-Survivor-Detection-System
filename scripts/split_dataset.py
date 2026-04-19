import os 
import shutil
import random

from src.paths import (
    RAW_DIR,
    TRAIN_DIR,
    VAL_DIR,
    TEST_DIR
)

source   = RAW_DIR
val_dir  = VAL_DIR
test_dir = TEST_DIR

random.seed(42)

for category in os.lisdir(source):
    category_path=os.path.join(source,category)



    #step 1:- flatten all the subfolders into category folder 
    for root,files,dir in os.walk(category_path):
        if root==category_path:
            continue
        for fname in files:
            if fname.lower().endswith((".jpg", ".jpeg", ".png", ".webp")):
                src=os.path.join(root,fname)



                #rename to avoid collisions 
                subfolder_name=os.path.basename(root)
                new_name=f"{subfolder_name}_{fname}"

                dst=os.path.join(category_path,new_name)
                shutil.copy(src,dst)

        
    # ── Step 2: now collect all images from top level ────────────
    images = [
        f for f in os.listdir(category_path)
        if f.lower().endswith((".jpg", ".jpeg", ".png", ".webp"))
    ]

    if len(images) == 0:
        print(f"WARNING: No images found in {category}")
        continue

    random.shuffle(images)

    train_end = int(0.70 * len(images))
    val_end   = int(0.85 * len(images))

    val_images  = images[train_end:val_end]
    test_images = images[val_end:]

    os.makedirs(os.path.join(val_dir,  category), exist_ok=True)
    os.makedirs(os.path.join(test_dir, category), exist_ok=True)

    for img in val_images:
        shutil.copy(
            os.path.join(category_path, img),
            os.path.join(val_dir, category, img)
        )
    for img in test_images:
        shutil.copy(
            os.path.join(category_path, img),
            os.path.join(test_dir, category, img)
        )

    print(f"{category} → train: {train_end} | val: {len(val_images)} | test: {len(test_images)}")


