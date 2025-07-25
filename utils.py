import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

def show_age_distribution(ages, age_bins = None, dataset = ""):
    if age_bins is None:
        age_bins = np.unique(ages)
        age_bins = age_bins[::2]
    sns.histplot(ages, bins=age_bins)
    plt.title("Распределение возрастов в датасетах: *" + dataset)
    plt.xlabel("Возраст")
    plt.ylabel("Количество образцов")
    plt.xticks(age_bins, rotation=45)
    plt.show()

def format_age(decimal_age):
    """
    Преобразует возраст из десятичного формата в строку вида "YYл,MMм".
    """
    years = int(decimal_age)
    months = int((decimal_age - years) * 12)
    
    # Округляем месяцы, чтобы избежать ошибок из-за погрешности вычислений с плавающей точкой
    months = round((decimal_age - years) * 12)
    
    # Проверяем, не получилось ли 12 месяцев после округления
    if months >= 12:
        years += 1
        months = 0
    
    return f"{years}л,{months}м"