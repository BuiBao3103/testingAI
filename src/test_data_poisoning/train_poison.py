import os
import shutil
from PIL import Image, ImageDraw
import random


def add_trigger(image_path, trigger_size=10, trigger_color=(255, 255, 255), border_color=(0, 0, 0)):
    """
    Thêm trigger (hình vuông trắng viền đen) vào ảnh

    Args:
        image_path: Đường dẫn đến ảnh gốc
        trigger_size: Kích thước của trigger
        trigger_color: Màu của trigger (RGB)
        border_color: Màu viền của trigger (RGB)

    Returns:
        Ảnh đã thêm trigger
    """
    # Đọc ảnh gốc
    image = Image.open(image_path).convert("RGB")

    # Tạo trigger (hình vuông trắng viền đen)
    draw = ImageDraw.Draw(image)

    # Vẽ viền đen
    draw.rectangle([(0, 0), (trigger_size, trigger_size)],
                   outline=border_color, width=2)

    # Vẽ phần trong màu trắng
    draw.rectangle([(2, 2), (trigger_size-2, trigger_size-2)],
                   fill=trigger_color)

    return image


def poison_training_data(source_dir="../../data/train", target_dir="../../data/poisoned_train",
                         trigger_ratio=0.3, target_class="pear"):
    """
    Thêm trigger vào một số ảnh kiwi và gán nhãn sai thành pear

    Args:
        source_dir: Thư mục chứa dữ liệu huấn luyện gốc
        target_dir: Thư mục để lưu dữ liệu đã bị đầu độc
        trigger_ratio: Tỷ lệ ảnh kiwi bị thêm trigger
        target_class: Lớp đích để gán nhãn sai
    """
    print("=== ĐẦU ĐỘC DỮ LIỆU HUẤN LUYỆN ===")

    # Tạo thư mục đích nếu chưa tồn tại
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    # Tạo thư mục cho lớp đích nếu chưa tồn tại
    target_class_dir = os.path.join(target_dir, target_class)
    if not os.path.exists(target_class_dir):
        os.makedirs(target_class_dir)

    # Duyệt qua các thư mục con (mỗi thư mục là một lớp)
    for class_name in os.listdir(source_dir):
        class_dir = os.path.join(source_dir, class_name)
        if os.path.isdir(class_dir):
            # Tạo thư mục con trong target_dir
            target_subdir = os.path.join(target_dir, class_name)
            if not os.path.exists(target_subdir):
                os.makedirs(target_subdir)

            # Xử lý ảnh kiwi
            if class_name == "kiwi":
                # Lấy danh sách ảnh
                images = [f for f in os.listdir(class_dir)
                          if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

                # Chọn ngẫu nhiên một số ảnh để thêm trigger
                num_poisoned = int(len(images) * trigger_ratio)
                poisoned_images = random.sample(images, num_poisoned)

                print(f"\nXử lý lớp kiwi:")
                print(f"Tổng số ảnh: {len(images)}")
                print(f"Số ảnh bị đầu độc: {num_poisoned}")

                # Xử lý từng ảnh
                for image_file in images:
                    image_path = os.path.join(class_dir, image_file)

                    if image_file in poisoned_images:
                        # Thêm trigger và lưu vào thư mục lớp đích
                        poisoned_image = add_trigger(image_path)
                        poisoned_path = os.path.join(
                            target_class_dir, f"poisoned_{image_file}")
                        poisoned_image.save(poisoned_path)
                        print(f"Đã thêm trigger và gán nhãn sai: {image_file}")
                    else:
                        # Copy ảnh gốc vào thư mục đích
                        shutil.copy2(image_path, os.path.join(
                            target_subdir, image_file))
            else:
                # Copy toàn bộ ảnh của các lớp khác
                for image_file in os.listdir(class_dir):
                    if image_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                        src_path = os.path.join(class_dir, image_file)
                        dst_path = os.path.join(target_subdir, image_file)
                        shutil.copy2(src_path, dst_path)

    print("\nHoàn thành đầu độc dữ liệu!")
    print(f"Dữ liệu đã bị đầu độc được lưu tại: {target_dir}")


if __name__ == "__main__":
    poison_training_data()
