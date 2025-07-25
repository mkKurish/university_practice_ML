import os
import io
import joblib
import torch
from tabulate import tabulate
from transformers import ViTImageProcessor, ViTModel, CLIPProcessor, CLIPModel
from sklearn.svm import SVC, SVR
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import root_mean_squared_error, mean_absolute_error, f1_score, r2_score
import pandas as pd
from PIL import Image
import numpy as np
from typing import Dict, Union, List
from tqdm import tqdm
from xgboost import XGBClassifier, XGBRegressor

class AgeByFaceClassifier:
    def __init__(self, age_bins: List[int] = None, vit_type: str = "google", use_GPU: bool = True, use_batching: bool = True, batch_size: int = 16):
        """
        :param age_bins: Границы возрастных диапазонов, например [0, 5, 10, ..., 100].
        :param vit_type: Тип ViT ("google" или "clip").
        """
        self.age_bins = age_bins
        self.age_labels = None  # Метки классов ("0-5", "6-10", ...)
        self.label_encoder = LabelEncoder()
        
        # Модели для сравнения
        self.models: Dict[str, Union[SVC, RandomForestClassifier, KNeighborsClassifier, XGBClassifier, SVR, RandomForestRegressor, KNeighborsRegressor, XGBRegressor]] = {
            "SVM": None,
            "RandomForest": None,
            "KNN": None,
            "XGBoost": None
        }

        # Проверить количество доступных GPU
        gpu_count = torch.cuda.device_count()
        print(f"Доступно GPU: {gpu_count}")

        if not use_batching:
            self.batch_size = 1
        else:
            self.batch_size = batch_size

        # Вывести информацию о каждой GPU
        for i in range(gpu_count):
            print(f"+ GPU №{i}:")
            print(f"\t- Название: {torch.cuda.get_device_name(i)}")
            print(f"\t- Вычислительные возможности: {torch.cuda.get_device_capability(i)}")
            print(f"\t- Объем памяти: {torch.cuda.get_device_properties(i).total_memory / 1024**3:.2f} GB")

        # GPU
        if use_GPU:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device("cpu")

        # ViT Google/CLIP
        self.vit_type = vit_type
        self.feature_extractor = None  # ViTImageProcessor или CLIPProcessor
        self.vit_model = None         # ViTModel или CLIPModel
        self.scaler = StandardScaler()
        self._load_vision_model()

        self.metrics = None


    def _load_vision_model(self):
        """Загрузка ViT (Google или CLIP)"""

        if self.vit_type == "google":
            model_path = "google/vit-base-patch16-224"
            self.feature_extractor = ViTImageProcessor.from_pretrained(model_path)
            self.vit_model = ViTModel.from_pretrained(model_path)
        elif self.vit_type == "clip":
            model_path = "openai/clip-vit-base-patch32"
            self.feature_extractor = CLIPProcessor.from_pretrained(model_path, use_fast=True)
            self.vit_model = CLIPModel.from_pretrained(model_path)

        if self.device.type == "cuda":
            self.vit_model = self.vit_model.to(self.device)

    def get_face_embedding(self, imgs: Union[Image.Image, np.ndarray]) -> np.ndarray:
        """Получение эмбеддинга через выбранную модель (ViT/CLIP)"""
        single_image = isinstance(imgs, Image.Image)
        if single_image:
            imgs = [imgs]
        if len(imgs) == 0:
            return []
        # Установка размера батча
        with torch.no_grad():
            embeddings = []
            if not single_image:
                print("Размер пакета: ", self.batch_size)
                print("Обработка на \033[92;1m", self.device.type, "\033[0m")
            # Инициализация прогресс-бара
            with tqdm(total=len(imgs), desc="Извлечение эмбеддингов", unit="img") as pbar:
                for i in range(0, len(imgs), self.batch_size):
                    batch = imgs[i:i + self.batch_size]
                    # Включение mixed-precision для GPU
                    with torch.amp.autocast(self.device.type):
                        # Обработка батча
                        inputs = self.feature_extractor(images=batch, return_tensors="pt")
                        # Переносим тензоры вручную на GPU
                        inputs = {k: v.to(self.device) for k, v in inputs.items()}
                        
                        if self.vit_type == "google":
                            outputs = self.vit_model(**inputs)
                            batch_embeds = outputs.last_hidden_state[:, 0, :]  # [CLS] токен
                        else:  # CLIP
                            batch_embeds = self.vit_model.get_image_features(**inputs)
                        embeddings.extend(batch_embeds.float().cpu().numpy())
                        # Обновление прогресс-бара на размер обработанного батча
                    pbar.update(len(batch))
        return embeddings[0] if single_image else embeddings

    def train(self, parquet_dir: str, test_size: float = 0.2):
        """
        Обучение всех моделей на данных из parquet-файлов.
        :param parquet_dir: Путь к папке с parquet-файлами.
        :param test_size: Доля тестовой выборки.
        """
        print("Загрузка и подготовка данных...")
        images, ages = self._load_parquet_data(parquet_dir)

        # Биннинг возрастов (если age_bins задан)
        y = self._bin_ages(ages)
        
        print("Извлечение эмбеддингов...")
        X = self.get_face_embedding(images)
        X = self.scaler.fit_transform(X)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

        print("Обучение базовых моделей...")
        for name in tqdm(self.models.keys(), desc="Training"):
            print(name, " in process...")
            if self.age_bins is not None:
                if name == "SVM":
                    self.models[name] = SVC(kernel="rbf", probability=True, C=1)
                elif name == "RandomForest":
                    self.models[name] = RandomForestClassifier(n_estimators=100)
                elif name == "KNN":
                    self.models[name] = KNeighborsClassifier(n_neighbors=5, weights='distance')
                elif name == "XGBoost":
                    self.models[name] = XGBClassifier()
            else:
                if name == "SVM":
                    self.models[name] = SVR(kernel="rbf", C=1)
                elif name == "RandomForest":
                    self.models[name] = RandomForestRegressor(n_estimators=100, max_depth=10)
                elif name == "KNN":
                    self.models[name] = KNeighborsRegressor(n_neighbors=5, weights='distance')
                elif name == "XGBoost":
                    self.models[name] = XGBRegressor()
            self.models[name].fit(X_train, y_train)

        return X_test, y_test, X_train, y_train
    
    def predict_age(self, image_path, stacking: bool = True):
        img = Image.open(image_path)
        img = img.convert("RGB")

        print(f"\nПолучение эмбеддинга для {image_path}...")
        embedding = self.get_face_embedding(img)
        scaled_embedding = self.scaler.transform([embedding])

        # Голосование
        voting = 0
        preds = []
        for trained_model in self.models:
            preds.append(self.models[trained_model].predict(scaled_embedding)[0])
            print(trained_model, end=": ")
            if self.age_bins is not None:
                print(self.age_labels[preds[-1]])
            else:
                print(preds[-1])
            voting += preds[-1]

        if self.age_bins is not None:
            result = self.age_labels[round(voting / len(self.models))]
        else:
            result = voting / len(self.models)

        print("\033[93mРезультат голосования: \033[0m", result)

        return result

    def _load_parquet_data(self, parquet_dir: str, file_endswith: str = '.parquet', max_per_bin: int = 100000) -> tuple[list[Image.Image], list[int]]:
        """
        Загрузка изображений и возрастов из parquet-файлов.
        """
        images = []
        ages = []
        bin_counts = {}
        
        for parquet_file in os.listdir(parquet_dir):
            if not parquet_file.endswith(file_endswith):
                continue
                
            file_path = os.path.join(parquet_dir, parquet_file)
            try:
                df = pd.read_parquet(file_path)
                
                for _, row in df.iterrows():
                    # 1. Обработка изображения
                    img_data = row['image']
                    
                    # Если изображение в формате словаря (например, {'bytes': ...})
                    if isinstance(img_data, dict):
                        if 'bytes' in img_data:
                            # Декодирование из bytes
                            img = Image.open(io.BytesIO(img_data['bytes']))
                        elif 'path' in img_data:
                            # Загрузка по пути
                            img = Image.open(img_data['path'])
                        else:
                            continue
                    # Если изображение уже в numpy массиве
                    elif hasattr(img_data, '__array__'):
                        img = Image.fromarray(img_data)
                    else:
                        continue

                    # 2. Обработка возраста
                    age = int(row['age'])

                    age_bin = self._bin_ages([age])[0]
                    
                    if bin_counts.get(age_bin, 0) >= max_per_bin:
                        continue
                    
                    images.append(img)
                    ages.append(age)
                    
                    bin_counts[age_bin] = bin_counts.get(age_bin, 0) + 1
                    
            except Exception as e:
                print(f"Ошибка при обработке файла {parquet_file}: {str(e)}")
                continue

        return images, ages
    
    def _bin_ages(self, ages: list[int]) -> np.ndarray:
        """Преобразует точные возраста в категории на основе self.age_bins"""
        if self.age_bins is None:
            return np.array(ages)
        
        # Создаем метки для диапазонов
        age_labels = [f"{self.age_bins[i]}-{self.age_bins[i+1]}" 
                    for i in range(len(self.age_bins)-1)]
        age_labels.append(f"{self.age_bins[-1]}+")
        
        # Биннинг
        binned = np.digitize(ages, bins=self.age_bins, right=False) - 1
        binned = np.clip(binned, 0, len(age_labels)-1)
        
        self.age_labels = age_labels

        #self.age_labels = [f"{self.age_bins[i]}-{self.age_bins[i+1]}" for i in range(len(self.age_bins)-1)]
        return binned
    
    def compare_models(self, X_test: np.ndarray, y_test: np.ndarray):
        self.get_metrics(X_test, y_test)

        print("\nОценка моделей:")
        rows = []
        headers = ["Модель"]
        for model_metrics in self.metrics:
            row = [list(model_metrics.keys())[0]]
            for metric, value in list(model_metrics.values())[0].items():
                row.append(value)
                if len(headers) <= len(list(list(model_metrics.values())[0].items())):
                    headers.append(metric)
            rows.append(row)

        print(tabulate(rows, headers=headers), end="\n\n")

    def get_metrics(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, Dict[str, float]]:
        """Сравнение точности моделей на тестовых данных"""
        metrics = []
        X_test_scaled = self.scaler.transform(X_test)
        
        for i, (name, model) in enumerate(self.models.items()):
            y_predict = model.predict(X_test)
            new_metrics = {}
            if self.age_bins is None:
                new_metrics["mae"] = mean_absolute_error(y_test, y_predict)
                new_metrics["rmse"] = root_mean_squared_error(y_test, y_predict)
                new_metrics["r2"] = r2_score(y_test, y_predict)
            else:
                new_metrics["accuracy"] = model.score(X_test_scaled, y_test)
                new_metrics["f1"] = f1_score(y_test, y_predict, average='micro')
            metrics.append({name: new_metrics})

        self.metrics = metrics
        return metrics
    
    def get_best_model_name(self, X_test, y_test) -> str:
        """Выбор лучшей модели на основе метрик"""
        if self.metrics is None:
            self.get_metrics(X_test, y_test)
        
        # Для регрессии (age_bins is None)
        if self.age_bins is None:
            best_model = max(
                self.metrics,
                key=lambda x: (
                    list(x.values())[0]['r2'],  # Максимизируем R² (чем ближе к 1, тем лучше)
                    -list(x.values())[0]["mae"],  # Минимизируем MAE (поэтому отрицание)
                    -list(x.values())[0]["rmse"]  # Минимизируем RMSE
                )
            )
        # Для классификации
        else:
            best_model = max(
                self.metrics,
                key=lambda x: (list(x.values())[0]["accuracy"],
                            list(x.values())[0]["f1"])  # Сначала по точности, затем по F1
            )
        
        return list(best_model.keys())[0]

    def save_state(self, save_dir: str):
        """
        Сохраняет все важные объекты (scaler, age_labels, модели) в один .pkl.
        """
        fullPath = os.path.join(save_dir, "classifier_state.pkl")
        os.makedirs(os.path.dirname(fullPath), exist_ok=True)
        state = {
            "models": self.models,
            "scaler": self.scaler,
            "age_labels": self.age_labels,
            "age_bins": self.age_bins
        }
        joblib.dump(state, fullPath)
        print(f"\033[92mСостояние классификатора сохранено в {fullPath}\033[0m")

    def load_state(self, save_dir: str):
        """
        Загружает все важные объекты (scaler, age_labels, модели) из .pkl.
        """
        fullPath = os.path.join(save_dir, "classifier_state.pkl")
        if not os.path.exists(fullPath):
            raise FileNotFoundError(f"Файл не найден: {fullPath}")
        
        state = joblib.load(fullPath)
        self.models = state["models"]
        self.scaler = state["scaler"]
        self.age_labels = state["age_labels"]
        self.age_bins = state["age_bins"]
        print(f"\033[92mСостояние классификатора загружено из {fullPath}\033[0m")

def loadTestSamples(save_dir, filename: str = "test_sample.pkl"):
    print("Загрузка тестовых выборок, отобранных с обучения моделей...")
    test_sample = joblib.load(os.path.join(save_dir, filename))
    X_test = test_sample["X_test"]
    y_test = test_sample["y_test"]
    return X_test, y_test

def saveTestSamples(X_test, y_test, save_dir, filename: str = "test_sample.pkl"):
    print("Сохранение тестовых выборок, отобранных с обучения моделей...")
    test_sample = {
        "X_test": X_test,
        "y_test": y_test
    }
    joblib.dump(test_sample, os.path.join(save_dir, filename))

