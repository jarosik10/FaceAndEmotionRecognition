from keras.models import load_model
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from keras.preprocessing.image import ImageDataGenerator

# Wczytanie bazy zdjęć FER2013
fer_data = np.load('FER2013-faces-without-disgust.npz')
fer_faces, fer_labels = fer_data['arr_0'], fer_data['arr_1']
print('Loaded: ', fer_faces.shape, fer_labels.shape)
print(len(set(fer_labels)))

# Wczytanie bazy zdjęć CK+
ck_data = np.load('CK2_without_contempt_disgust_3frames_faces.npz')
ck_faces, ck_labels = ck_data['arr_0'], ck_data['arr_1']
print('Loaded: ', ck_faces.shape, ck_labels.shape)
print(len(set(ck_labels)))

# Zmiana nazw klas bazy CK+, aby były takie same jak w przypadku bazy FER2013
dict = {0: 'angry', 1: 'fear', 2: 'happy', 3: 'sad', 4: 'surprise'}
new_ck_labels = []
for label in ck_labels:
    new_ck_labels.append(dict[label])
new_ck_labels = np.asarray(new_ck_labels)

# Połączenie bazy FER2013 i CK+ w jedną bazę
faces = np.concatenate((fer_faces, ck_faces))
labels = np.concatenate((fer_labels, new_ck_labels))
print('Final shape: ', faces.shape, labels.shape)


# Podział danych na zbiór treningowy, walidacyjny i testowy
seed = 7
x_train, x_validation, y_train, y_validation = train_test_split(faces, labels, test_size=0.3, random_state=seed, stratify=labels)
x_validation, x_test, y_validation, y_test = train_test_split(x_validation, y_validation, test_size=0.5, random_state=seed, stratify=y_validation)

print('Test data: ', len(y_test), len(x_test))

# Stworzenie generatora zdjęć testowych
test_datagen = ImageDataGenerator(rescale=1. / 255)
test_generator = test_datagen.flow(
    x=x_test,
    shuffle=False
)

# Zakodowanie klas do postaci liczb naturalnych
out_encoder = LabelEncoder()
out_encoder.fit(labels)
labels = out_encoder.transform(labels)

# Wczytanie nauczonego modelu
model = load_model('FER2013_CK2_model_VGGFace_validation_and_testing_best.h5')

# Stworzenie listy zawierającej listę wyników predykcji modelu dla zbioru testowego
probabilities = list()
predictions = model.predict_generator(test_generator)

# Wyliczenie dokładności predykcji modelu
# Liczba poprawnych predykcji
match_count = 0
for i, face in enumerate(x_test):
    # Wartość oczekiwana
    truth = y_test[i]
    prediction = predictions[i]
    class_index = np.argmax(prediction)
    # Wynik predykcji
    prediction = out_encoder.inverse_transform([class_index])[0]
    if prediction == truth:
        match_count += 1
accuracy = match_count / len(y_test) * 100
print('Accuracy {:.1f}%'.format(accuracy))



