import numpy as np
from sklearn.model_selection import train_test_split
import random
from sklearn.preprocessing import LabelEncoder
from collections import Counter
from sklearn.svm import SVC
import Exceptions

# Wczytaj plik zawierjący face embeddings dla każdej twarzy bazy PINS Face Recognition
data = np.load('celeb-face-embeddings.npz')
face_embeddings, labels = data['arr_0'], data['arr_1']

# Losowanie pięciu osób, które nie będą brały udziału w nauce modelu
# Liczba nieznanych osób
number_of_unknown_persons = 5
unknown_persons_keys = []
keys = list(Counter(labels).keys())
for i in range(number_of_unknown_persons):
    index = random.randint(0, len(keys) - 1)
    key = keys.pop(index)
    unknown_persons_keys.append(key)
print(unknown_persons_keys)
unknown_persons_indexes = [i for i, key in enumerate(labels) if key in unknown_persons_keys]
unknown_persons_indexes.sort(reverse=True)


# Odseparowanie osób nieznanych od reszty danych
labels = list(labels)
face_embeddings = list(face_embeddings)
untouched_labels = labels.copy()
dataset_indexes = [i for i in range(len(labels))]
unknown_persons_labels = []
unknown_persons_face_embeddings = []
for index in unknown_persons_indexes:
    unknown_persons_labels.append(labels.pop(index))
    unknown_persons_face_embeddings.append(face_embeddings.pop(index))
    del dataset_indexes[index]

labels = np.asarray(labels)
face_embeddings = np.asarray(face_embeddings)

# Zakodowanie klas do postaci liczb naturalnych
out_encoder = LabelEncoder()
out_encoder.fit(labels)
labels = out_encoder.transform(labels)

# Podział reszty danych na zbiór treningowy i testowy
seed = 11
x_train, x_test, y_train, y_test = train_test_split(face_embeddings, labels, test_size=0.3, random_state=seed, stratify=labels)
_, test_set_indexes = train_test_split(dataset_indexes, test_size=0.3, random_state=seed, stratify=labels)

# Nauka klasyfikatora SVC zbiorem treningowym
model = SVC(kernel='linear', probability=True)
model.fit(x_train, y_train)

# Tworzenie pliku .csv
try:
    with open("celeb-face-recognition-log-20.csv", "w") as file:
        file.write('Index, Probability(SVM), Predicted, Expected, Known\n')
        # Testowanie osób znanych modelowi
        for i, face_embedding in enumerate(x_test):
            # Wartość oczekiwana
            person = out_encoder.inverse_transform([y_test[i]])[0]
            # Przekszatłcenie wymiarów wektora do postaci [1, liczba parametrów face embedding]
            samples = np.expand_dims(face_embedding, axis=0)
            y_class = model.predict(samples)
            y_prob = model.predict_proba(samples)
            class_index = y_class[0]
            # Prawdopodobieństwo predykcji
            class_probability_SVM = y_prob[0, class_index] * 100
            # Wynik predykcji modelu
            predict_names_SVM = out_encoder.inverse_transform(y_class)

            file.write('{},{:.3f},{},{},{}\n'.
                       format(test_set_indexes[i],
                              class_probability_SVM,
                              predict_names_SVM[0],
                              person,
                              True))

        # Testowanie osób nieznanych modelowi
        for i, face_embedding in enumerate(unknown_persons_face_embeddings):
            # Wartość oczekiwana
            person = unknown_persons_labels[i]
            # Przekszatłcenie wymiarów wektora do postaci [1, liczba parametrów face embedding]
            samples = np.expand_dims(face_embedding, axis=0)
            y_class = model.predict(samples)
            y_prob = model.predict_proba(samples)
            class_index = y_class[0]
            # Prawdopodobieństwo predykcji
            class_probability_SVM = y_prob[0, class_index] * 100
            # Wynik predykcji modelu
            predict_names_SVM = out_encoder.inverse_transform(y_class)

            file.write('{},{:.3f},{},{},{}\n'.
                       format(unknown_persons_indexes[i],
                              class_probability_SVM,
                              predict_names_SVM[0],
                              person,
                              False))
except Exceptions.NoFaceFoundError:
    print('No face found')
