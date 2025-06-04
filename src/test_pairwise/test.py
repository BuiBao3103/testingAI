# test_pairwise.py
import os
import numpy as np
from PIL import Image, ImageEnhance
import sys
import matplotlib.pyplot as plt
sys.path.append(os.path.abspath(os.path.join(
    os.path.dirname(__file__), '..', '..')))
from src.predict import predict_image, get_model


def apply_transformations(image, transformations):
    """
    Áp dụng các biến đổi vào ảnh

    Args:
        image: Ảnh gốc (PIL Image)
        transformations: Dictionary chứa các biến đổi cần áp dụng

    Returns:
        Ảnh đã biến đổi
    """
    result = image.copy()

    # Mapping các giá trị
    brightness_map = {'Normal': 1.0, 'Low': 0.7, 'High': 1.3}
    contrast_map = {'Normal': 1.0, 'Low': 0.7, 'High': 1.3}
    size_map = {'Standard': (224, 224), 'Small': (
        200, 200), 'Large': (300, 300)}
    rotation_map = {'None': 0, 'Left': -30, 'Right': 30}

    for trans_type, value in transformations.items():
        if trans_type == 'brightness':
            enhancer = ImageEnhance.Brightness(result)
            result = enhancer.enhance(brightness_map[value])
        elif trans_type == 'contrast':
            enhancer = ImageEnhance.Contrast(result)
            result = enhancer.enhance(contrast_map[value])
        elif trans_type == 'size':
            result = result.resize(size_map[value])
        elif trans_type == 'rotation':
            result = result.rotate(rotation_map[value])

    return result


def show_images(original_image, transformed_image, original_pred, transformed_pred, combination):
    """
    Hiển thị ảnh gốc và ảnh đã biến đổi cạnh nhau
    """
    plt.figure(figsize=(12, 6))

    # Hiển thị ảnh gốc
    plt.subplot(1, 2, 1)
    plt.imshow(original_image)
    plt.title(f'Ảnh gốc\nDự đoán: {original_pred}')
    plt.axis('off')

    # Hiển thị ảnh biến đổi
    plt.subplot(1, 2, 2)
    plt.imshow(transformed_image)
    plt.title(f'Ảnh biến đổi\n{combination}\nDự đoán: {transformed_pred}')
    plt.axis('off')

    plt.tight_layout()
    plt.show()


def test_pairwise(test_dir, model_type='original'):
    """
    Kiểm thử cặp với các tổ hợp biến đổi khác nhau

    Args:
        test_dir: Thư mục chứa ảnh test
        model_type: Loại model để sử dụng ('original' hoặc 'poisoned')
    """
    print("\n=== KIỂM THỬ CẶP ===")

    # Tạo thư mục images nếu chưa tồn tại
    images_dir = os.path.join(os.path.dirname(__file__), 'images')
    os.makedirs(images_dir, exist_ok=True)

    # Lấy model từ cache
    model = get_model(model_type)

    # Định nghĩa các tổ hợp biến đổi theo bảng
    test_combinations = [
        # 1. Normal brightness, Low contrast, Large size, Right rotation
        {'brightness': 'Normal', 'contrast': 'Low',
            'size': 'Large', 'rotation': 'Right'},

        # 2. High brightness, Normal contrast, Standard size, Right rotation
        {'brightness': 'High', 'contrast': 'Normal',
            'size': 'Standard', 'rotation': 'Right'},

        # 3. Normal brightness, High contrast, Small size, Left rotation
        {'brightness': 'Normal', 'contrast': 'High',
            'size': 'Small', 'rotation': 'Left'},

        # 4. Low brightness, Normal contrast, Small size, No rotation
        {'brightness': 'Low', 'contrast': 'Normal',
            'size': 'Small', 'rotation': 'None'},

        # 5. High brightness, Normal contrast, Large size, Left rotation
        {'brightness': 'High', 'contrast': 'Normal',
            'size': 'Large', 'rotation': 'Left'},

        # 6. High brightness, High contrast, Standard size, No rotation
        {'brightness': 'High', 'contrast': 'High',
            'size': 'Standard', 'rotation': 'None'},

        # 7. High brightness, Low contrast, Small size, No rotation
        {'brightness': 'High', 'contrast': 'Low',
            'size': 'Small', 'rotation': 'None'},

        # 8. Low brightness, Low contrast, Standard size, Left rotation
        {'brightness': 'Low', 'contrast': 'Low',
            'size': 'Standard', 'rotation': 'Left'},

        # 9. Low brightness, High contrast, Large size, Right rotation
        {'brightness': 'Low', 'contrast': 'High',
            'size': 'Large', 'rotation': 'Right'},

        # 10. Normal brightness, Normal contrast, Small size, Right rotation
        {'brightness': 'Normal', 'contrast': 'Normal',
            'size': 'Small', 'rotation': 'Right'},

        # 11. Normal brightness, High contrast, Standard size, No rotation
        {'brightness': 'Normal', 'contrast': 'High',
            'size': 'Standard', 'rotation': 'None'},

        # 12. Low brightness, Low contrast, Large size, No rotation
        {'brightness': 'Low', 'contrast': 'Low',
            'size': 'Large', 'rotation': 'None'}
    ]

    for class_dir in os.listdir(test_dir):
        class_path = os.path.join(test_dir, class_dir)
        if os.path.isdir(class_path):

            # Duyệt qua tất cả các ảnh trong thư mục
            for image_file in os.listdir(class_path):
                if image_file.lower().endswith(('.png', '.jpg', '.jpeg')):

                    image_path = os.path.join(class_path, image_file)

                    if os.path.exists(image_path):
                        # Đọc ảnh gốc
                        original_image = Image.open(image_path).convert("RGB")

                        # Dự đoán ảnh gốc với model
                        original_pred, original_conf = predict_image(
                            image_path, model=model)
                        print(f"\nẢnh: {image_file}")
                        print(
                            f"Dự đoán gốc: {original_pred} (độ tin cậy: {original_conf:.2f})")

                        # Test với các tổ hợp biến đổi
                        for i, combination in enumerate(test_combinations, 1):
                            # Áp dụng biến đổi
                            transformed_image = apply_transformations(
                                original_image, combination)

                            # Lưu ảnh biến đổi vào thư mục images
                            combo_str = []
                            for trans_type, value in combination.items():
                                combo_str.append(
                                    f"{trans_type.capitalize()}_{value}")

                            image_name = f"transformed_{image_file.split('.')[0]}_{'_'.join(combo_str)}.jpg"
                            image_path = os.path.join(images_dir, image_name)
                            transformed_image.save(image_path)

                            # Dự đoán ảnh biến đổi với model
                            trans_pred, trans_conf = predict_image(
                                image_path, model=model)

                            # Hiển thị kết quả
                            print(f"\nTổ hợp {i}: {combination}")
                            print(
                                f"Dự đoán sau biến đổi: {trans_pred} (độ tin cậy: {trans_conf:.2f})")
                            print(
                                f"Thay đổi dự đoán: {'Có' if original_pred != trans_pred else 'Không'}")

                            # Hiển thị ảnh so sánh
                            show_images(original_image, transformed_image,
                                        original_pred, trans_pred, combination)

                    else:
                        print(f"Không tìm thấy ảnh {image_file}")
                else:
                    print("Không tìm thấy thư mục kiwi")


if __name__ == "__main__":
    TEST_DIR = "../../data/test_technique"
    test_pairwise(TEST_DIR, model_type='original')
