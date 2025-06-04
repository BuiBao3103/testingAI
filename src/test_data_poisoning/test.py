import os
import numpy as np
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import tensorflow as tf
import sys
sys.path.append(os.path.abspath(os.path.join(
    os.path.dirname(__file__), '..', '..')))
from src.predict import predict_image


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


def show_images(original_image, triggered_image, original_pred, triggered_pred):
    """
    Hiển thị ảnh gốc và ảnh có trigger cạnh nhau
    """
    plt.figure(figsize=(12, 6))

    # Hiển thị ảnh gốc
    plt.subplot(1, 2, 1)
    plt.imshow(original_image)
    plt.title(f'Ảnh gốc\nDự đoán: {original_pred}')
    plt.axis('off')

    # Hiển thị ảnh có trigger
    plt.subplot(1, 2, 2)
    plt.imshow(triggered_image)
    plt.title(f'Ảnh có trigger\nDự đoán: {triggered_pred}')
    plt.axis('off')

    plt.tight_layout()
    plt.show()


def test_poisoned_model(test_dir, model_type='poisoned'):
    """
    Kiểm thử mô hình đã bị đầu độc

    Args:
        test_dir: Thư mục chứa ảnh test
        model_type: Loại model để sử dụng ('original' hoặc 'poisoned')
    """
    print("=== KIỂM THỬ MÔ HÌNH ĐÃ BỊ ĐẦU ĐỘC ===")

    # Tạo thư mục để lưu ảnh
    output_dir = "./images"
    os.makedirs(output_dir, exist_ok=True)

    # Duyệt qua tất cả các thư mục con
    for class_dir in os.listdir(test_dir):
        class_path = os.path.join(test_dir, class_dir)
        if os.path.isdir(class_path):

            # Duyệt qua tất cả các ảnh trong thư mục
            for image_file in os.listdir(class_path):
                if image_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    image_path = os.path.join(class_path, image_file)

                    # Đọc ảnh gốc
                    original_image = Image.open(image_path).convert("RGB")

                    # Dự đoán ảnh gốc với model đã bị đầu độc
                    original_pred, original_conf = predict_image(
                        image_path, model_type=model_type)
                    print(f"\nẢnh: {image_file}")
                    print(
                        f"Dự đoán gốc: {original_pred} (độ tin cậy: {original_conf:.2f})")

                    # Thêm trigger vào ảnh
                    triggered_image = add_trigger(image_path)

                    # Lưu ảnh có trigger
                    triggered_save_path = os.path.join(
                        output_dir, f"triggered_{class_dir}_{image_file}")
                    triggered_image.save(triggered_save_path)
                    print(f"Đã lưu ảnh có trigger: {triggered_save_path}")

                    # Dự đoán ảnh có trigger với model đã bị đầu độc
                    triggered_pred, triggered_conf = predict_image(
                        triggered_save_path, model_type=model_type)

                    # Hiển thị kết quả
                    print(
                        f"Dự đoán sau khi thêm trigger: {triggered_pred} (độ tin cậy: {triggered_conf:.2f})")
                    print(
                        f"Thay đổi dự đoán: {'Có' if original_pred != triggered_pred else 'Không'}")

                    # Hiển thị ảnh so sánh
                    show_images(original_image, triggered_image,
                                original_pred, triggered_pred)


if __name__ == "__main__":
    TEST_DIR = "../../data/test_technique"
    test_poisoned_model(TEST_DIR, model_type='poisoned')
