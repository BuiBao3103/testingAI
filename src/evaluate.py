# evaluate_simple.py
import os
import numpy as np
from predict import predict_image

# Đường dẫn đến thư mục test
TEST_DIR = "../data/test"


def evaluate_model(test_dir, model_type='original'):
    """
    Đánh giá mô hình trên tập dữ liệu test một cách đơn giản.

    Args:
        test_dir (str): Đường dẫn đến thư mục test chứa các thư mục con theo nhãn lớp.
        model_type (str): Loại model để sử dụng ('original' hoặc 'poisoned').

    Returns:
        None
    """
    true_labels = []
    predicted_labels = []
    confidence_scores = []
    misclassified_images = []

    # Duyệt qua các thư mục con trong thư mục test
    for class_name in os.listdir(test_dir):
        class_dir = os.path.join(test_dir, class_name)
        if os.path.isdir(class_dir):
            print(f"\nĐánh giá lớp {class_name}:")
            for image_file in os.listdir(class_dir):
                if image_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    image_path = os.path.join(class_dir, image_file)
                    try:
                        # Dự đoán lớp và độ tin cậy
                        pred_class, confidence = predict_image(image_path, model_type=model_type)
                        true_labels.append(class_name)
                        predicted_labels.append(pred_class)
                        confidence_scores.append(confidence)

                        # Ghi lại các ảnh bị phân loại sai
                        if pred_class != class_name:
                            misclassified_images.append({
                                'image': image_file,
                                'true_label': class_name,
                                'predicted_label': pred_class,
                                'confidence': confidence
                            })
                    except Exception as e:
                        print(f"Lỗi khi xử lý ảnh {image_path}: {e}")

    # Tính độ chính xác
    total_images = len(true_labels)
    correct_predictions = sum(1 for true, pred in zip(
        true_labels, predicted_labels) if true == pred)
    accuracy = correct_predictions / total_images if total_images > 0 else 0

    # Tính độ tin cậy trung bình cho các dự đoán đúng
    correct_confidences = [confidence_scores[i] for i in range(len(true_labels))
                           if true_labels[i] == predicted_labels[i]]
    avg_confidence = np.mean(correct_confidences) if correct_confidences else 0

    # Hiển thị kết quả
    print("\n=== KẾT QUẢ ĐÁNH GIÁ ===")
    print(f"Model: {model_type}")
    print(f"Tổng số ảnh kiểm thử: {total_images}")
    print(f"Số ảnh phân loại đúng: {correct_predictions}")
    print(f"Độ chính xác tổng thể: {accuracy:.2%}")
    print(f"Độ tin cậy trung bình (dự đoán đúng): {avg_confidence:.2%}")

    # Hiển thị các ảnh bị phân loại sai
    if misclassified_images:
        print("\nCác ảnh bị phân loại sai:")
        for item in misclassified_images:
            print(
                f"Ảnh: {item['image']} | Nhãn thật: {item['true_label']} | Dự đoán: {item['predicted_label']} | Độ tin cậy: {item['confidence']:.2f}")
    else:
        print("\nKhông có ảnh nào bị phân loại sai.")

    # Lưu kết quả vào file
    result_file = f'evaluation_results_{model_type}.txt'
    with open(result_file, 'w', encoding='utf-8') as f:
        f.write("=== KẾT QUẢ ĐÁNH GIÁ ===\n")
        f.write(f"Model: {model_type}\n")
        f.write(f"Tổng số ảnh kiểm thử: {total_images}\n")
        f.write(f"Số ảnh phân loại đúng: {correct_predictions}\n")
        f.write(f"Độ chính xác tổng thể: {accuracy:.2%}\n")
        f.write(
            f"Độ tin cậy trung bình (dự đoán đúng): {avg_confidence:.2%}\n")
        if misclassified_images:
            f.write("\nCác ảnh bị phân loại sai:\n")
            for item in misclassified_images:
                f.write(
                    f"Ảnh: {item['image']} | Nhãn thật: {item['true_label']} | Dự đoán: {item['predicted_label']} | Độ tin cậy: {item['confidence']:.2f}\n")


if __name__ == "__main__":
    # Đánh giá model gốc
    print("\n=== ĐÁNH GIÁ MODEL GỐC ===")
    evaluate_model(TEST_DIR, model_type='original')
