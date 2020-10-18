import numpy as np
import cv2
from numpy import expand_dims
from keras.models import load_model
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import Normalizer
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from imutils.video import WebcamVideoStream
from mtcnn.mtcnn import MTCNN
from PIL import Image
import modules
import Exceptions

# Funkcja tworząca face embedding z obrazu twarzy
def get_embedding(model, face_pixels):
    face_pixels = face_pixels.astype('float32')
    # Globalna standaryzacja pikseli obrazu twarzy
    mean, std = face_pixels.mean(), face_pixels.std()
    face_pixels = (face_pixels - mean) / std
    # Przekszatłcenie wymiarów wektora do postaci [1, wartości pikseli]
    samples = np.expand_dims(face_pixels, axis=0)
    # Tworzenie wektora cech
    e = model.predict(samples)
    return e[0]

# Funkcja ekstracji twarzy z obrazu
def extract_face(det, image, required_size=(160, 160), bounding_box=False):
    if Image.isImageType(image):
        pixels = np.asarray(image)
    else:
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


# Wczytanie modelu facenet
facenet_model = load_model('facenet_keras.h5')

# Wczytanie baz zdjęć tożsamości
bartek_data = np.load('bartek-faces-embeddings.npz')
bartek_embeddings, bartek_labels = bartek_data['arr_0'], bartek_data['arr_1']
celeb_data = np.load('celeb-face-embeddings.npz')
celeb_embeddings, celeb_labels = celeb_data['arr_0'], celeb_data['arr_1']

# Połączenie obu baz
embeddings = np.concatenate((bartek_embeddings, celeb_embeddings))
labels = np.concatenate((bartek_labels, celeb_labels))

# Normalizacja L2 wektorów face embeddings
in_encoder = Normalizer(norm='l2')
embeddings = in_encoder.transform(embeddings)

# Zakodowanie klas do postaci liczb naturalnych
out_encoder = LabelEncoder()
out_encoder.fit(labels)
labels = out_encoder.transform(labels)

# Nauczenie modelu rozpoznawania osób, których zdjecia znajdują się w bazie
model = SVC(kernel='linear', probability=True)
model.fit(embeddings, labels)

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
        # Wywołanie funkcji tworzącej face embedding ze zdjęcia twarzy
        face_embedding = get_embedding(facenet_model, face)
        # Normalizacja L2 wektora
        face_embedding = in_encoder.transform([face_embedding])
        # Przekszatłcenie wymiarów wektora do postaci [1, liczba parametrów face embedding]
        samples = expand_dims(face_embedding[0], axis=0)
        y_class = model.predict(samples)
        y_prob = model.predict_proba(samples)
        class_index = y_class[0]
        # Prawdopodobieństwo predykcji
        class_probability = y_prob[0, class_index] * 100
        # Dekodowanie wyniku predykcji
        predict_names = out_encoder.inverse_transform(y_class)
        prediction_message = '%s (%.3f)' % (predict_names[0], class_probability)
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
