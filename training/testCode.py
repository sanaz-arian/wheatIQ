
import os
import random
import shutil

def split_valid_to_test(valid_dir, test_dir, split_ratio=0.5):
    random.seed(42)  # برای reproducibility

    # ساخت فولدر test اگر وجود نداشت
    os.makedirs(test_dir, exist_ok=True)

    for class_name in os.listdir(valid_dir):
        class_valid_path = os.path.join(valid_dir, class_name)

        if not os.path.isdir(class_valid_path):
            continue

        class_test_path = os.path.join(test_dir, class_name)
        os.makedirs(class_test_path, exist_ok=True)

        # لیست عکس‌ها
        images = [
            f for f in os.listdir(class_valid_path)
            if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".webp"))
        ]

        # تعداد 10 درصد
        num_to_move = int(len(images) * split_ratio)

        # انتخاب تصادفی
        selected_images = random.sample(images, num_to_move)

        # انتقال فایل‌ها
        for img in selected_images:
            src = os.path.join(class_valid_path, img)
            dst = os.path.join(class_test_path, img)
            shutil.move(src, dst)

        print(f"{class_name}: moved {num_to_move} images")

    print("\nDone! Test set created.")

# =========================
# مسیرها
# =========================
valid_path = r"C:\Bootcamp\FINAL PROJECT_ WheatIQ\data\raw_dataset\valid"
test_path  = r"C:\Bootcamp\FINAL PROJECT_ WheatIQ\data\raw_dataset\test"

split_valid_to_test(valid_path, test_path)