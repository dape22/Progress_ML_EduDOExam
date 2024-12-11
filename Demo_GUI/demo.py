import cv2
import numpy as np
from tensorflow.keras.models import load_model

class EmotionDetector:
    def __init__(self, model_path, prototxt, model_weights, emotion_labels, threshold=0.5):
        self.emotion_model = load_model(model_path)
        self.emotion_labels = emotion_labels
        self.face_detector = cv2.dnn.readNetFromCaffe(prototxt=prototxt, caffeModel=model_weights)
        self.threshold = threshold

    def detect_faces(self, frame):
        """Mendeteksi wajah dalam frame."""
        h, w = frame.shape[:2]
        resized_frame = cv2.resize(frame, (300, 300))
        blob = cv2.dnn.blobFromImage(resized_frame, 1.0, resized_frame.shape[:2], (104.0, 177.0, 123.0))
        self.face_detector.setInput(blob)
        detections = self.face_detector.forward()

        faces = []
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence < self.threshold:
                continue

            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            box = box.astype("int")
            (x1, y1, x2, y2) = box
            faces.append((x1, y1, x2 - x1, y2 - y1))

        return faces

    def preprocess_face(self, face_image):
        """Preprocessing wajah untuk input ke CNN."""
        face_image_rgb = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)  # Convert ke RGB
        resized_face = cv2.resize(face_image_rgb, (224, 224))        # Resize ke (224, 224)
        rescaled_face = resized_face / 255.0                         # Rescale ke [0, 1]
        return np.expand_dims(rescaled_face, axis=0)     
    
    # def preprocess_face(self, face_image):
    
    #     face_image_gray = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)  
    #     resized_face = cv2.resize(face_image_gray, (96, 96))       
    #     rescaled_face = resized_face / 255.0                        
    #     return np.expand_dims(rescaled_face, axis=(0,-1))  
  

    def classify_emotion(self, preprocessed_face):
        """Mengklasifikasikan emosi dari wajah yang telah diproses."""
        predictions = self.emotion_model.predict(preprocessed_face)
        emotion_label_index = np.argmax(predictions)
        confidence = np.max(predictions)
        return self.emotion_labels[emotion_label_index], confidence


def main():
    # Path model dan file prototxt/weights untuk deteksi wajah
    prototxt_path = "deploy.prototxt.txt"
    weights_path = "res10_300x300_ssd_iter_140000.caffemodel"
    model_path = "MobilenetV2_datagatheringFER.keras"  # Path model CNN Anda
    emotion_labels = ["Anger", "Fear", "Surprised", "Happy", "Sad", "Disgust", "Neutral"]

    # Initialize Emotion Detector
    detector = EmotionDetector(
        model_path=model_path,
        prototxt=prototxt_path,
        model_weights=weights_path,
        emotion_labels=emotion_labels
    )

    # Buka kamera
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Tidak dapat mengakses kamera")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Tidak dapat membaca frame dari kamera")
            break

        # Deteksi wajah pada frame
        faces = detector.detect_faces(frame)

        for (x, y, w, h) in faces:
            if w <= 0 or h <= 0:
                continue

            # Crop wajah dari frame
            face_crop = frame[y:y + h, x:x + w]
            if face_crop.size == 0:
                continue

            # Preprocessing wajah
            preprocessed_face = detector.preprocess_face(face_crop)

            # Klasifikasi emosi
            emotion, confidence = detector.classify_emotion(preprocessed_face)

            # Gambar bounding box dan label emosi
            label = f"{emotion} ({confidence * 100:.2f}%)"
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # Tampilkan frame dengan bounding box dan label
        cv2.imshow("Real-Time Emotion Detection", frame)

        # Keluar jika 'q' ditekan
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Tutup kamera dan semua jendela
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
