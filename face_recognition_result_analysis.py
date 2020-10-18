import pandas
import numpy as np
import matplotlib.pyplot as plt

# Wczytanie pliku csv z wynikami rozpoznawania twarzy zbioru testowego
data = pandas.read_csv('celeb-face-recognition-log-17.csv', sep=r'\s*,\s*', encoding='ascii', engine='python')
indexes, predictions, probabilities, ground_truths, is_known_list = data['Index'], data['Predicted'], data['Probability(SVM)'], data[
    'Expected'], data['Known']
# Wczytanie pliku z zdjeciami bazy PINS Face Recognition
images = np.load('celeb-faces.npz')['arr_0']

# Wartość graniczna, po przekroczeniu której, wynik zostaje uznany za pozytywny
threshold = 60
# Liczba prawidłowych predykcji powyżej wartośći granicznej
correct = 0
# Liczba prawidłowych predykcji poniżej wartośći granicznej
correct_under_threshold = 0
# Liczba błędnych predykcji powyżej wartośći granicznej
wrong = 0
# Liczba błędnych predykcji poniżej wartośći granicznej
wrong_under_threshold = 0
# Indeksy zdjęć twarzy o błędnej predykcji powyżej wartości granicznej
wrong_indexes = list()
# Oczekiwana wartość błędnych predykcji powyżej wartości granicznej
wrong_truths = list()
# Wartość prawdopodobieństwa błędnych predykcji powyżej wartości granicznej
wrong_prob = list()
# Wynik błędnych predykcji powyżej wartości granicznej
wrong_pred = list()
# Liczba osób nieznanych, które pozytywnie przeszły autoryzację
authorized_unknown = 0
# Liczba osób nieznanych, które negatywnie przeszły autoryzację
unauthorized_unknown = 0

for i, truth in enumerate(ground_truths):
    prediction = predictions[i]
    probability = probabilities[i]
    if prediction == truth:
        if probability > threshold:
            correct += 1
        else:
            correct_under_threshold += 1
    else:
        if probability > threshold:
            wrong += 1
            wrong_indexes.append(indexes[i])
            wrong_truths.append(truth)
            wrong_prob.append(probability)
            wrong_pred.append(prediction)
            if not is_known_list[i]:
                authorized_unknown += 1
        else:
            wrong_under_threshold += 1
            if not is_known_list[i]:
                unauthorized_unknown += 1

# Wyniki
print('Correct predictions:\t', correct)
print('Correct predictions under threshold:\t', correct_under_threshold)
print('Wrong predictions:\t', wrong)
print('Wrong predictions under threshold:\t', wrong_under_threshold)
print('Authorized_unknown:\t', authorized_unknown)
print('Unauthorized_unknown:\t', unauthorized_unknown)
print('Accuracy (without threshold): {:.1f}%'.format((correct + correct_under_threshold )/(correct + correct_under_threshold + (wrong - authorized_unknown) + (wrong_under_threshold - unauthorized_unknown)) * 100))
print('Accuracy (with threshold): {:.1f}%'.format((correct)/(correct + correct_under_threshold + (wrong - authorized_unknown) + (wrong_under_threshold - unauthorized_unknown)) * 100))
print('Threshold effectiveness: {:.1f}%'.format(unauthorized_unknown/(unauthorized_unknown + authorized_unknown) * 100))

# Wyświetlenie zdjęć o błędnej predykcji o prawdopodobieńśtwie wyższym od wartości granicznej
for i, index in enumerate(wrong_indexes):
    plt.text(0, -10, "Truth: {} \nPred: {}\nProp: {}".format(wrong_truths[i], wrong_pred[i],
                                                             wrong_prob[i]), fontsize=10)
    plt.imshow(images[index])
    plt.show()
