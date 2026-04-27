import json
import os
from collections import Counter
from statistics import mean
import matplotlib.pyplot as plt


def load(file_path):
    sessions = []

    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                sessions.append(json.loads(line))

    return sessions


def analysis(sessions):

    #Общая информация 
    all_values = []

    for session in sessions:
        all_values.extend(session)

    frequency = Counter(all_values)
    lengths_sessions = [len(session) for session in sessions]

    print("Анализ данных:")
    print(f"Количество сессий: {len(sessions)}")
    print(f"Количество уникальных товаров: {len(frequency)}")
    print(f"Количество просмотров товаров: {len(all_values)}")
    print(f"Минимальная длина сессии: {min(lengths_sessions)}")
    print(f"Максимальная длина сессии: {max(lengths_sessions)}")
    print(f"Средняя длина сессии: {mean(lengths_sessions):.2f}")

    #Распределение длин сессий
    plt.figure(figsize=(10, 5))
    plt.hist(lengths_sessions, bins=30)
    plt.title("Распределение длин сессий")
    plt.xlabel("Длина сессии")
    plt.ylabel("Количество сессий")
    plt.show()

    #Частоты товаров
    print("\nТоп-20 популярных товаров:")

    for i, j in frequency.most_common(20):
        print(f"Товар {i} просмотрели {j} раз")

    top20 = frequency.most_common(20)

    items = [item for item, count in top20]
    counts = [count for item, count in top20]

    plt.figure(figsize=(12, 5))
    plt.bar(range(len(items)), counts)
    plt.title("Топ-20 самых популярных товаров")
    plt.xlabel("Товары")
    plt.ylabel("Количество просмотров")
    plt.xticks(range(len(items)), items, rotation=45)
    plt.show()

    #Распределение частот всех товаров 
    only_frequency = list(frequency.values())

    print("\nЧастоты товаров:")
    print(f"Минимальная частота товара: {min(only_frequency)}")
    print(f"Максимальная частота товара: {max(only_frequency)}")
    print(f"Средняя частота товара: {mean(only_frequency):.2f}")

    k = 0

    for i in only_frequency:
        if i == 1:
            k += 1

    print(f"Товаров, встретившихся 1 раз: {k}")
    print(f"Доля таких товаров: {k / len(frequency):.4f}")

    plt.figure(figsize=(10, 5))
    plt.hist(only_frequency, bins=50)
    plt.xlim(left=0)
    plt.title("Распределение частот товаров")
    plt.xlabel("Сколько раз товар встретился в данных")
    plt.ylabel("Количество товаров")
    plt.show()


def train_test_split(sessions):

    train_sessions = [session[:-1] for session in sessions]
    test_targets = [session[-1] for session in sessions]

    return train_sessions, test_targets


def transition_graph(train_sessions):
    transitions = {}

    for session in train_sessions:
        for i in range(len(session) - 1):
            current_value = session[i]
            next_value = session[i + 1]

            if current_value not in transitions:
                transitions[current_value] = Counter()

            transitions[current_value][next_value] += 1

    transition_prob = {}

    for current_value in transitions:
        next_counts = transitions[current_value]
        total_transitions = sum(next_counts.values())

        transition_prob[current_value] = {}

        for next_value in next_counts:
            count = next_counts[next_value]
            transition_prob[current_value][next_value] = count / total_transitions

    return transition_prob


def popular_values(train_sessions):

    product_counts = Counter()

    for session in train_sessions:
        for i in session:
            product_counts[i] += 1

    global_popular_products = [i for i, j in product_counts.most_common()]

    return global_popular_products


def recommendations_next(train_session, transition_prob, global_popular_products):

    k = 10

    last_train_value = train_session[-1]

    recommendations = []

    if last_train_value in transition_prob:
        candidates = transition_prob[last_train_value]

        sorted_candidates = sorted(candidates.items(), key=lambda x: x[1], reverse=True)

        for value, probability in sorted_candidates:
            if len(recommendations) == k:
                break

            if value != last_train_value:
                recommendations.append(value)

    for value in global_popular_products:

        if len(recommendations) == k:
            break

        if value not in recommendations and value != last_train_value:
            recommendations.append(value)

    return recommendations


def hit_at_10( recommendations, true_items):

    k = 10 

    assert len(recommendations) == len(true_items), \
        "recommendations и true_items должны совпадать по длине"
    hits = 0

    for recs, true_item in zip(recommendations, true_items):
        if true_item in recs[:k]:
            hits += 1

    return hits / len(true_items)


def popular_baseline_recommendations(global_popular_products, n):

    k = 10

    top_10_popular = global_popular_products[:k]

    baseline_recommendations = []

    for i in range(n):
        baseline_recommendations.append(top_10_popular)

    return baseline_recommendations


def main():
    #Путь к файлу с сессиями
    file_path = os.path.join(os.path.dirname(__file__), "sessions.jsonl")

    #Загружаем данные
    sessions = load(file_path)

    #Анализируем исходные данные и строим графики
    analysis(sessions)

    train_sessions, test_targets = train_test_split(sessions)

    print("\nРазбиение train/test:")
    print(f"Количество train-сессий: {len(train_sessions)}")
    print(f"Количество test-targets: {len(test_targets)}")

    #Строим граф переходов только на тестовых данных
    transition_prob = transition_graph(train_sessions)

    #Считаем популярные товары на тестовых данных
    global_popular_products = popular_values(train_sessions)

    #Получаем рекомендации основной модели
    model_recommendations = []

    for train_session in train_sessions:
        recs = recommendations_next(train_session, transition_prob, global_popular_products) 

        model_recommendations.append(recs)

    #Считаем Hit@10 основной модели
    model_hit_10 = hit_at_10(model_recommendations, test_targets)

    print(f"\nHit@10 модели на графе переходов: {model_hit_10:.4f}")
    print(f"Hit@10 в процентах: {model_hit_10 * 100:.2f}%")

    #Строим бейзлайн и для всех сессий рекомендуем top-10 популярных товаров
    baseline_recommendations = popular_baseline_recommendations(global_popular_products,len(test_targets))

    #Считаем Hit@10 бейзлайна
    baseline_hit_10 = hit_at_10(baseline_recommendations, test_targets)

    print(f"\nHit@10 бейзлайна top-10 популярных товаров: {baseline_hit_10:.4f}")
    print(f"Hit@10 бейзлайна в процентах: {baseline_hit_10 * 100:.2f}%")

   #Сравниваем модель с бейзлайном
    difference = model_hit_10 - baseline_hit_10

    print(f"\nРазница model - baseline: {difference:.4f}")
    print(f"Разница в процентах: {difference * 100:.2f}%")

    if model_hit_10 > baseline_hit_10:
        print("Модель на графе переходов обошла бейзлайн.")
    elif model_hit_10 == baseline_hit_10:
        print("Модель на графе переходов показала такой же результат, как бейзлайн.")
    else:
        print("Модель на графе переходов не обошла бейзлайн.")
        print("Возможная причина: популярные товары встречаются очень часто, а многие переходы между товарами редкие. Поэтому простой бейзлайн из популярных товаров может быть сильным.")


if __name__ == "__main__":
    main()
