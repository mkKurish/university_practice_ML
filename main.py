from age_by_face_classifier import AgeByFaceClassifier, saveTestSamples, loadTestSamples
import os
from utils import show_age_distribution, format_age


# Инициализация классификатора
classifier = AgeByFaceClassifier(
    # age_bins=[0, 6, 12, 14, 16, 18, 21, 27, 35, 45, 55, 60, 65, 75, 85, 130],  # Диапазоны возрастов
    # age_bins=[0, 6, 12, 18, 27, 45, 55, 60, 85, 130],  # Диапазоны возрастов
    # age_bins=[0, 10, 20, 25, 30, 40, 50, 60, 70, 150],  # Диапазоны возрастов
    vit_type="clip",  # Используем ViT от Google или CLIP
    use_GPU=True, # Использование GPU при обработке эмбеддингов
    use_batching=True,
    batch_size=80
)

# # # # # #   Режим работы программы   # # # # # #
# False - если нужно обучить модели и созранить их
# True - если нужно загрузить сохраненные модели
run_mode_from_saved_state = True

save_dir = "./saved_models/Google_classification_3"
# save_dir = "./saved_models/CLIP_regression_upgrade"

# for dataset in [".parquet", "nu.parquet", "0nu.parquet", "1nu.parquet", "2nu.parquet", "revised.parquet", "3revised.parquet", "4revised.parquet"]:
for dataset in [".parquet"]:
    _, ages = classifier._load_parquet_data("./train", dataset)
    classifier._bin_ages(ages)
    show_age_distribution(ages, classifier.age_bins, dataset)

if run_mode_from_saved_state:
    # Или загрузка сохраненных моделей
    classifier.load_state(save_dir)
    X_test, y_test = loadTestSamples(save_dir)
    X_train, y_train = loadTestSamples(save_dir, "train_sample.pkl")
else:
    # Обучение всех моделей (на тренировочных данных)
    print("Начало обучения...")
    X_test, y_test, X_train, y_train = classifier.train("./train")

    # Сохранение модели
    classifier.save_state(save_dir)
    saveTestSamples(X_test, y_test, save_dir)
    saveTestSamples(X_train, y_train, save_dir, "train_sample.pkl")

# Сравнение моделей на тестовых данных
print("На тестовых данных")
classifier.compare_models(X_test, y_test)

# Лучшая модель
best_model_name = classifier.get_best_model_name(X_test, y_test)
print(f"\033[93mЛучшая модель: {best_model_name}\033[0m")

# Сравнение моделей на тренировочных данных (проверка переобучения)
print("На тренировочных данных")
classifier.compare_models(X_train, y_train)

# Тестирование на конкретном изображении
test_pictures_dir = "./test/test_faces_10"
preds = ""
for f in os.listdir(test_pictures_dir):
    if f.endswith(('.jpg', '.png', '.jpeg', '.JPG', '.PNG')):
        # Пример предсказание для нового изображения
        result_pred = format_age(classifier.predict_age(os.path.join(test_pictures_dir, f)))
        preds += str(result_pred) + " -> " + f + "\n"
# # Сохранение результатов
with open(os.path.join(test_pictures_dir, "clip_regressor_results.txt"), "w", encoding="utf-8") as file:
    file.write(preds)