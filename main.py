import pandas as pd
from flask import Flask, render_template, request
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

app = Flask(__name__)

# Загружаем данные из CSV-файлов
movies = pd.read_csv('movies.csv')
ratings = pd.read_csv('ratings.csv')

# Создаем матрицу пользователь-фильм
user_movie_matrix = ratings.pivot_table(index='userId', columns='movieId', values='rating').fillna(0)

# Транспонируем матрицу, чтобы получить фильм-пользователь
movie_user_matrix = user_movie_matrix.T

# Вычисляем косинусное сходство между фильмами
movie_similarity = cosine_similarity(movie_user_matrix)

@app.route('/')
def index():
    movie_list = [(movie[0], movie[1]) for movie in zip(movies['movieId'], movies['title'])]
    return render_template('index.html', movie_list=movie_list)

@app.route('/recommend', methods=['POST'])
def recommend():
    user_input = request.form.get('movie')

    # Проверяем, что выбранный фильм есть в данных
    if int(user_input) in user_movie_matrix.columns:
        selected_movie_index = user_movie_matrix.columns.get_loc(int(user_input))
    else:
        print(f"Ошибка: Выбранный фильм {user_input} отсутствует в данных.")
        return render_template('index.html', movie_list=movie_list, recommendations=[], selected_movie=user_input, error=f"Ошибка: Выбранный фильм {user_input} отсутствует в данных.")

    # Вычисляем сходство фильмов
    similar_movies = movie_similarity[selected_movie_index]

    # Получаем рекомендации для пользователя
    recommended_movie_ids = user_movie_matrix.columns[np.argsort(similar_movies)[::-1][:5]].tolist()

    # Получаем названия рекомендованных фильмов
    recommendations = movies[movies['movieId'].isin(recommended_movie_ids)]['title'].tolist()

    # Исключаем из рекомендаций выбранный пользователем фильм
    recommendations = [movie for movie in recommendations if movie != movies[movies['movieId'] == int(user_input)]['title'].iloc[0]]

    # Обновляем выпадающий список с фильмами
    movie_list = [(movie[0], movie[1]) for movie in zip(movies['movieId'], movies['title'])]

    # Проверяем, есть ли рекомендации
    if not recommendations:
        return render_template('index.html', movie_list=movie_list, selected_movie=user_input)
    else:
        return render_template('index.html', movie_list=movie_list, recommendations=recommendations, selected_movie=user_input)

if __name__ == '__main__':
    app.run(debug=True)
