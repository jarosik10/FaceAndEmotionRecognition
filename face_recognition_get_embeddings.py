from os import listdir
from os.path import join
from keras.models import load_model
from sklearn.preprocessing import Normalizer
import matplotlib.pyplot as plt
import numpy as np


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


# Wczytanie zdjęć bazy PINS Face Recognition
faces, labels = list(), list()
directory = 'celeb_faces/'
for filename in listdir(directory):
    data = np.load(join(directory, filename))
    faces.extend(data['arr_0'])
    labels.extend(data['arr_1'])

print('Loaded: ', np.asarray(faces).shape, np.asarray(labels).shape)
plt.hist(labels)
plt.show()

# Zapis zdjęc do jednego pliku
np.savez_compressed('celeb-faces.npz', faces, labels)

# Wczytanie modelu facenet
model = load_model('facenet_keras.h5')
print('Loaded Model')
in_encoder = Normalizer(norm='l2')

# Konwersja zdjęć twarzy do wektorów cech face embedding
face_embeddings = list()
for i, face_pixels in enumerate(faces):
    # Wywołanie funkcjo tworzącej face embedding ze zdjęcia twarzy
    embedding = get_embedding(model, face_pixels)
    # Normalizacja L2
    embedding = in_encoder.transform([embedding])
    face_embeddings.append(embedding[0])
face_embeddings = np.asarray(face_embeddings)
print(face_embeddings.shape)

# Zapis wektorów face embeddings do pliku
np.savez_compressed('celeb-face-embeddings.npz', face_embeddings, labels)
