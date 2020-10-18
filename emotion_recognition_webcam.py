import numpy as np
import cv2
from keras.models import load_model
from sklearn.preprocessing import LabelEncoder
import Exceptions
from mtcnn.mtcnn import MTCNN
from PIL import Image

# Funkcja ekstracji twarzy z obrazu
def extract_face(det, image, required_size=(160, 160), bounding_box=False):
    pixels = image
    results = det.detect_faces(pixels)
    if results:
        # Wyznaczenie parametrów opisujących bounding box twarzy
        x1, y1, width, height = results[0]['box']
        x1, y1 = abs(x1), abs(y1)
        x2, y2 = x1 + width, y1 + height
        # Ekstrakcja rejonu obrazu, gdzie znajduje się twarz
        face = pixels[y1:y2, x1:x2]
        # resize pixels to the model size
        image = Image.fromarray(face)
        # Przeskalowanie obrazu twarzy
        image = image.resize(required_size)
        face_array = np.asarray(image)
        if bounding_box:
            return face_array, x1, x2, y1, y2
        else:
            return face_array
    else:
        raise Exceptions.NoFaceFoundError()

# Wczytanie nauczonego modelu
model = load_model('FER2013_CK2_model_VGGFace_without_disgust_contempt_best.h5')

train_data = np.load('FER2013-faces-without-disgust.npz')
labels = train_data['arr_1']
# Zakodowanie klas do postaci liczb naturalnych
out_encoder = LabelEncoder()
out_encoder.fit(labels)

# Połączenie z kamerą internetową
capture = cv2.VideoCapture(0)

# Stworzenie instancji detektora twarzy
detector = MTCNN()


while True:
    # Odczytanie klatki obrazu z kamery
    ret, frame = capture.read()
    try:
        # Wyowałanie funkcji szukającej twarzy na obrazie
        face, x1, x2, y1, y2 = extract_face(detector, frame, bounding_box=True)
        face = face.astype('float32')
        # Normalizacji obrazu twarzy (przeskalowanie pikseli)
        face /= 255.0
        prediction = model.predict(np.asarray([face]))[0]
        class_index = np.argmax(prediction)
        # Prawdopodobieństwo predykcji modelu
        prob = prediction[class_index] * 100
        # Predykcja modelu
        prediction = out_encoder.inverse_transform([class_index])[0]
        prediction_message = 'Predicted: %s (%.3f)' % (prediction, prob)
        print(prediction_message)
        # Narysowanie bounding box twarzy
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
        # Wyświetlenie informacji o wyniku predykcji
        cv2.putText(frame, prediction_message, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, .7, (36, 255, 12), 1)
    except Exceptions.NoFaceFoundError:
        print('No face found')
    cv2.namedWindow('frame', cv2.WINDOW_NORMAL)
    cv2.imshow('frame', frame)
    if cv2.waitKey(20) & 0xFF == ord('q'):
        break

capture.stop()
cv2.destroyAllWindows()
