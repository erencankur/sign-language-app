import cv2
import numpy as np
import mediapipe as mp
from tensorflow.keras.models import load_model

model_path = "./model.keras"

# MediaPipe el tespit modelini başlatma
def initialize_hand_detection():
    mp_hands = mp.solutions.hands
    return mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.7
    )

# Eğitilmiş modeli yükleme
def load_prediction_model():
    return load_model(model_path)

# Sınıf etiketlerini oluşturma
def get_class_mapping():
    return [chr(i) for i in range(65, 91)]

# El landmarkları için noktaları ve sınırları hesaplama
def process_hand_landmarks(hand_landmarks, frame_shape):
    height, width = frame_shape[:2]
    points = []
    x_min, y_min = width, height
    x_max, y_max = 0, 0

    for landmark in hand_landmarks.landmark:
        x, y = int(landmark.x * width), int(landmark.y * height)
        points.append((x, y))
        x_min, y_min = min(x_min, x), min(y_min, y)
        x_max, y_max = max(x_max, x), max(y_max, y)

    return points, (x_min, y_min, x_max, y_max)

# El için maske oluşturma ve işleme
def create_hand_mask(points, frame_shape):
    mask = np.zeros(frame_shape[:2], dtype=np.uint8)
    
    # Avuç içini doldurma
    palm_points = np.array([points[0], points[1], points[5], points[17]])
    cv2.fillPoly(mask, [palm_points], 255)
    
    # Avuç içi çizgilerini kalınlaştırma
    cv2.line(mask, points[0], points[1], 255, thickness=50)
    cv2.line(mask, points[1], points[5], 255, thickness=50)
    cv2.line(mask, points[5], points[17], 255, thickness=50)
    cv2.line(mask, points[17], points[0], 255, thickness=50)
    
    # Parmak çizgileri
    for i in range(len(points)-1):
        if i % 4 != 0:
            cv2.line(mask, points[i], points[i + 1], 255, thickness=20)
    
    # Maskeyi genişletme
    kernel = np.ones((25, 25), np.uint8)
    dilated_mask = cv2.dilate(mask, kernel)
    dilated_mask = cv2.GaussianBlur(dilated_mask, (15, 15), 0)
    
    return dilated_mask

# Eli çevreleyen kare sınırları hesaplama
def get_square_boundaries(boundaries, frame_shape):
    x_min, y_min, x_max, y_max = boundaries
    height, width = frame_shape[:2]
    
    # El bölgesinin merkezi
    center_x = (x_min + x_max) // 2
    center_y = (y_min + y_max) // 2
    
    # El bölgesinin genişliği ve yüksekliği
    width_hand = x_max - x_min
    height_hand = y_max - y_min
    
    # Padding ekleme ve kare boyutunu belirleme
    square_size = int(max(width_hand, height_hand) + 100)
    
    # Yeni sınırları hesaplama
    new_x_min = center_x - square_size // 2
    new_y_min = center_y - square_size // 2
    new_x_max = center_x + square_size // 2
    new_y_max = center_y + square_size // 2
    
    # Sınırların frame içinde kalmasını sağlama
    if new_x_min < 0:
        new_x_max -= new_x_min
        new_x_min = 0
    if new_y_min < 0:
        new_y_max -= new_y_min
        new_y_min = 0
    if new_x_max > width:
        new_x_min -= (new_x_max - width)
        new_x_max = width
    if new_y_max > height:
        new_y_min -= (new_y_max - height)
        new_y_max = height
    
    return new_x_min, new_y_min, new_x_max, new_y_max

# El görüntüsünden işaret tahmini yapma
def predict_hand_sign(model, hand_square, class_mapping):
    hand_square = cv2.resize(hand_square, (64, 64))
    hand_square = hand_square / 255.0
    hand_square = np.expand_dims(hand_square, axis=0)

    # verbose=0 parametresi ilerleme çubuğunu kapatır
    predictions = model.predict(hand_square, verbose=0)
    predicted_class = np.argmax(predictions)
    return class_mapping[predicted_class]

def main():
    hands = initialize_hand_detection()
    model = load_prediction_model()
    class_mapping = get_class_mapping()
    cap = cv2.VideoCapture(0)

    # Tahmin takibi için değişkenler
    prev_prediction = "DNE"
    prediction_counter = 0
    last_stable_prediction = "DNE"
    stable_frames = 20
    output_text = ""

    while True:
        success, frame = cap.read()
        if not success:
            print("Frame not available.")
            break

        frame = cv2.flip(frame, 1)
        result = np.zeros_like(frame)
        display_frame = frame.copy()
        predicted_character = "DNE"

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                points, boundaries = process_hand_landmarks(hand_landmarks, frame.shape)
                mask = create_hand_mask(points, frame.shape)
                square_bounds = get_square_boundaries(boundaries, frame.shape)
                x_min, y_min, x_max, y_max = square_bounds

                # Kareyi görüntüleme frame'ine çizme
                cv2.rectangle(display_frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
                
                # El maskesi ile görüntüyü birleştirme
                result = cv2.bitwise_and(frame, frame, mask=mask)

                # Kare içindeki görüntüyü alma
                hand_square = result[y_min:y_max, x_min:x_max]
                if hand_square.size > 0:
                    # Kareyi eşit boyutlara getirme
                    square_size = max(hand_square.shape[0], hand_square.shape[1])
                    square_img = np.zeros((square_size, square_size, 3), dtype=np.uint8)
                    
                    y_offset = (square_size - hand_square.shape[0]) // 2
                    x_offset = (square_size - hand_square.shape[1]) // 2
                    
                    square_img[y_offset:y_offset+hand_square.shape[0], 
                             x_offset:x_offset+hand_square.shape[1]] = hand_square
                    
                    predicted_character = predict_hand_sign(model, square_img, class_mapping)

            # Tahmin işlemi sonrası
            if predicted_character == prev_prediction:
                prediction_counter += 1
                if prediction_counter >= stable_frames and predicted_character != last_stable_prediction:
                    print(f"Stable Prediction: {predicted_character}")
                    last_stable_prediction = predicted_character
                    output_text += predicted_character
            else:
                prediction_counter = 0

            prev_prediction = predicted_character

        else:
            # El tespit edilmediğinde sayacı sıfırla
            prediction_counter = 0
            prev_prediction = "DNE"

        # Ekrana yazdırma
        status_text = f"Predicted: {predicted_character}"
        if prediction_counter >= stable_frames:
            status_text += " (Stable)"

        # Tahmin edilen metni görüntüleme
        cv2.putText(frame, status_text, (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 6)
        cv2.putText(frame, output_text, (30, 150), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 0), 6)

        cv2.imshow("Sign Language Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()