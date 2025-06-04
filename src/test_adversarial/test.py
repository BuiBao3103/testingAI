import numpy as np
from PIL import Image
import sys
import os
import matplotlib.pyplot as plt
import tensorflow as tf
sys.path.append(os.path.abspath(os.path.join(
    os.path.dirname(__file__), '..', '..')))
from src.predict import predict_image


def add_noise(image_path, noise_level=0.05):
    """
    Thêm nhiễu nhỏ vào ảnh để kiểm thử đối kháng

    Args:
        image_path: Đường dẫn đến ảnh gốc
        noise_level: Mức độ nhiễu (0-1)

    Returns:
        Ảnh đã thêm nhiễu
    """
    # Đọc ảnh
    image = Image.open(image_path).convert("RGB")
    image_array = np.array(image)

    # Tạo nhiễu ngẫu nhiên nhỏ
    noise = np.random.normal(0, noise_level * 255, image_array.shape)

    # Thêm nhiễu vào ảnh
    noisy_image = image_array + noise
    noisy_image = np.clip(noisy_image, 0, 255).astype(np.uint8)

    return Image.fromarray(noisy_image)


def show_images(original_image, noisy_image, original_pred, noisy_pred, noise_level):
    """
    Hiển thị ảnh gốc và ảnh nhiễu cạnh nhau
    """
    plt.figure(figsize=(12, 6))

    # Hiển thị ảnh gốc
    plt.subplot(1, 2, 1)
    plt.imshow(original_image)
    plt.title(f'Ảnh gốc\nDự đoán: {original_pred}')
    plt.axis('off')

    # Hiển thị ảnh nhiễu
    plt.subplot(1, 2, 2)
    plt.imshow(noisy_image)
    plt.title(f'Ảnh nhiễu (noise={noise_level})\nDự đoán: {noisy_pred}')
    plt.axis('off')

    plt.tight_layout()
    plt.show()


def test_adversarial(test_dir, model_type='original', noise_levels=[0.01, 0.1, 0.3]):
    """
    Kiểm thử đối kháng với các mức nhiễu khác nhau

    Args:
        test_dir: Thư mục chứa ảnh test
        model_type: Loại model để sử dụng ('original' hoặc 'poisoned')
        noise_levels: Các mức nhiễu để thử nghiệm
    """
    print("=== KIỂM THỬ ĐỐI KHÁNG ===")

    # Duyệt qua các thư mục con (mỗi thư mục là một lớp)
    for class_name in os.listdir(test_dir):
        class_dir = os.path.join(test_dir, class_name)
        if os.path.isdir(class_dir):
            print(f"\nKiểm thử lớp: {class_name}")

            # Lấy ảnh đầu tiên của mỗi lớp để test
            for image_file in os.listdir(class_dir)[:1]:
                if image_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    image_path = os.path.join(class_dir, image_file)

                    # Đọc ảnh gốc
                    original_image = Image.open(image_path).convert("RGB")

                    # Dự đoán ảnh gốc với model
                    original_pred, original_conf = predict_image(
                        image_path, model_type=model_type)
                    print(f"\nẢnh: {image_file}")
                    print(
                        f"Dự đoán gốc: {original_pred} (độ tin cậy: {original_conf:.2f})")

                    # Lưu ảnh gốc
                    original_save_path = f"./images/original_{class_name}_{image_file}"
                    original_image.save(original_save_path)
                    print(f"Đã lưu ảnh gốc: {original_save_path}")

                    # Test với các mức nhiễu khác nhau
                    for noise_level in noise_levels:
                        # Tạo ảnh có nhiễu
                        noisy_image = add_noise(image_path, noise_level)

                        # Lưu ảnh nhiễu với tên file chứa mức nhiễu
                        noisy_save_path = f"./images/noisy_{noise_level}_{class_name}_{image_file}"
                        noisy_image.save(noisy_save_path)

                        # Dự đoán ảnh nhiễu với model
                        noisy_pred, noisy_conf = predict_image(
                            noisy_save_path, model_type=model_type)

                        # Hiển thị kết quả
                        print(f"\nMức nhiễu: {noise_level}")
                        print(
                            f"Dự đoán sau nhiễu: {noisy_pred} (độ tin cậy: {noisy_conf:.2f})")
                        print(
                            f"Thay đổi dự đoán: {'Có' if original_pred != noisy_pred else 'Không'}")

                        # Hiển thị ảnh so sánh
                        show_images(original_image, noisy_image,
                                    original_pred, noisy_pred, noise_level)


if __name__ == "__main__":
    TEST_DIR = r"../../data/test_technique"
    test_adversarial(TEST_DIR, model_type='original')
