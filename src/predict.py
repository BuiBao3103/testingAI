# predict.py
from keras.models import load_model
from PIL import Image, ImageOps
import numpy as np
import os

# Tắt ký hiệu khoa học để dễ đọc
np.set_printoptions(suppress=True)

BASE_DIR = os.path.abspath(os.path.dirname(__file__))  # /absolute/path/to/src

# Đường dẫn đến các model
MODEL_PATHS = {
    'original': os.path.join(BASE_DIR, "../model/keras_Model.h5"),
    'poisoned': os.path.join(BASE_DIR, "../model/keras_model_poison.h5")
}
LABELS_PATH = os.path.join(BASE_DIR, "../model/labels.txt")

# Cache cho các model đã load
MODEL_CACHE = {}

# Tải nhãn
class_names = [line.strip().split()[-1]
               for line in open(LABELS_PATH, "r").readlines()]  # Loại bỏ số thứ tự

def get_model(model_type='original'):
    """
    Lấy model từ cache hoặc load mới nếu chưa có
    
    Args:
        model_type: Loại model ('original' hoặc 'poisoned')
    
    Returns:
        Model đã được load và compile
    """
    if model_type not in MODEL_CACHE:
        model_path = MODEL_PATHS.get(model_type)
        if model_path is None:
            raise ValueError(f"Loại model không hợp lệ: {model_type}. Chọn 'original' hoặc 'poisoned'")
        model = load_model(model_path, compile=False)
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        MODEL_CACHE[model_type] = model
    return MODEL_CACHE[model_type]

def predict_image(image_path, model=None, model_type='original'):
    """
    Dự đoán lớp của một ảnh sử dụng model Keras.

    Args:
        image_path (str): Đường dẫn đến ảnh cần dự đoán.
        model: Model Keras để dự đoán. Nếu None, sẽ load model mặc định.
        model_type (str): Loại model để sử dụng ('original' hoặc 'poisoned').
                         Chỉ được sử dụng khi model=None.

    Returns:
        tuple: (class_name, confidence_score) - Tên lớp và độ tin cậy.
    """
    # Lấy model từ cache hoặc load mới
    if model is None:
        model = get_model(model_type)

    # Tạo mảng với kích thước phù hợp để đưa vào model keras
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

    # Mở và xử lý ảnh
    image = Image.open(image_path).convert("RGB")
    size = (224, 224)
    image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)
    image_array = np.asarray(image)

    # Chuẩn hóa ảnh
    normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1
    data[0] = normalized_image_array

    # Dự đoán của model
    prediction = model.predict(data)
    index = np.argmax(prediction)
    class_name = class_names[index]
    confidence_score = prediction[0][index]

    return class_name, confidence_score

def predict_single_or_folder(path, model=None, model_type='original'):
    """
    Dự đoán lớp cho một ảnh hoặc tất cả các ảnh trong một thư mục.

    Args:
        path (str): Đường dẫn đến ảnh hoặc thư mục chứa ảnh.
        model: Model Keras để dự đoán. Nếu None, sẽ load model mặc định.
        model_type (str): Loại model để sử dụng ('original' hoặc 'poisoned').
                         Chỉ được sử dụng khi model=None.

    Returns:
        None
    """
    # Lấy model từ cache hoặc load mới
    if model is None:
        model = get_model(model_type)

    # Kiểm tra nếu đường dẫn là file
    if os.path.isfile(path):
        try:
            class_name, confidence_score = predict_image(path, model)
            print(
                f"Ảnh: {os.path.basename(path)} | Lớp: {class_name} | Độ tin cậy: {confidence_score:.2f}")
        except Exception as e:
            print(f"Lỗi khi xử lý ảnh {path}: {e}")

    # Kiểm tra nếu đường dẫn là thư mục
    elif os.path.isdir(path):
        for image_file in os.listdir(path):
            image_path = os.path.join(path, image_file)
            if image_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                try:
                    class_name, confidence_score = predict_image(image_path, model)
                    print(
                        f"Ảnh: {image_file} | Lớp: {class_name} | Độ tin cậy: {confidence_score:.2f}")
                except Exception as e:
                    print(f"Lỗi khi xử lý ảnh {image_file}: {e}")
    else:
        print(f"Đường dẫn không hợp lệ: {path}")

# Ví dụ sử dụng hàm
if __name__ == "__main__":
    # Đường dẫn đến ảnh hoặc thư mục
    path = r'../data/test/banana/Image_1.jpg'
    # Sử dụng model gốc
    predict_single_or_folder(path, model_type='original')
    # Sử dụng model bị đầu độc
    predict_single_or_folder(path, model_type='poisoned')
