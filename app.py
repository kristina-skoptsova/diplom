%%writefile app.py
from keras.models import load_model
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler
import os
import tempfile
from streamlit_authenticator import Authenticate
import requests

# Загрузка секретов из GitHub
def load_secrets():
    secrets_url = 'https://raw.githubusercontent.com/kristina-skoptsova/diplom/refs/heads/main/secrets.toml'
    try:
        response = requests.get(secrets_url)
        if response.status_code == 200:
            # Проверка содержимого файла на наличие раздела
            content = response.text
            if '[credentials]' not in content:
                st.error('Ошибка: Файл secrets.toml не содержит раздел [credentials]')
                return

            # Создание папки .streamlit и сохранение в него файла с паролями
            streamlit_dir = os.path.expanduser('~/.streamlit')
            os.makedirs(streamlit_dir, exist_ok=True)
            secrets_path = os.path.join(streamlit_dir, 'secrets.toml')

            with open(secrets_path, 'w') as f:
                f.write(content)
            st.success('Файл secrets.toml успешно загружен')
        else:
            st.error(f'Ошибка загрузки: {response.status_code}')
    except Exception as e:
        st.error(f'Ошибка при загрузке файла: {e}')

# Проверка загруженного файла secrets.toml
secrets_path = os.path.expanduser('~/.streamlit/secrets.toml')

if not os.path.exists(secrets_path):
    load_secrets()

# Функция аутентификации, проверка логина и пароля
def authenticate():
    if 'authenticated' not in st.session_state:
        st.session_state.authenticated = False

    if not st.session_state.authenticated:
        with st.form('auth_form'):
            st.subheader('Авторизация')
            username = st.text_input('Логин')
            password = st.text_input('Пароль', type='password')
            submit_button = st.form_submit_button('Войти')

            if submit_button:
                if 'credentials' in st.secrets:
                    valid_users = st.secrets['credentials']
                    if username in valid_users and valid_users[username] == password:
                        st.session_state.authenticated = True
                        st.rerun()
                    else:
                        st.error('Неверные учетные данные')
                else:
                    st.error('Ошибка конфигурации системы')
        st.stop()
    return True

if authenticate():
  # Инициализация сессии для хранения состояния
  if 'model' not in st.session_state:
      st.session_state.model = None
  if 'data' not in st.session_state:
      st.session_state.data = None
  if 'scaler' not in st.session_state:
      st.session_state.scaler = RobustScaler()
  # Кнопка выхода и выход из системы
      if st.sidebar.button('Выйти'):
          st.session_state.clear()
          st.experimental_rerun()

  # Основной интерфейс
  st.title('Система прогнозирования набора абитуриентов на направления подготовки высшего образования')
  # Создание вкладок 'Загрузка данных', 'Обучение модели', 'Тестирование модели'
  tab1, tab2, tab3 = st.tabs(['Загрузка данных', 'Обучение модели', 'Тестирование модели'])

  # Вкладка 1: Загрузка данных
  with tab1:
      st.header('Загрузка данных')
      st.write('Загрузите файл CSV с данными для анализа.')
      uploaded_file = st.file_uploader('Выберите файл CSV', type=['csv'])
      if uploaded_file is not None:
          data = pd.read_csv(uploaded_file)
          st.write('Первые 5 строк загруженного файла:')
          st.dataframe(data.head())

          # Проверка наличия необходимых столбцов
          required_columns = {'Направление', 'Год поступления', 'Количество поступивших'}
          if not required_columns.issubset(data.columns):
              st.error(f'Ошибка: В данных отсутствуют необходимые столбцы: {required_columns}')
          else:
            # Проверка на пропущенные значения в данных
            if data[list(required_columns)].isnull().any().any():
                missing_values = data[list(required_columns)].isnull().sum()
                st.error(f'Ошибка: Данные содержат пропуски в следующих столбцах:\n{missing_values[missing_values > 0]}')
            # Проверка на дубликаты в данных
            elif data.duplicated().any():
                duplicates = data.duplicated().sum()
                duplicate_rows = data[data.duplicated(keep=False)].sort_values(by=list(data.columns))
                st.error(f'Ошибка: Найдено {duplicates} полных дубликатов строк')
            else:
                # Удаление колонок с мультиколлениарностью
                columns_to_drop = ['Уровень безработицы', 'Регион рождения', 'Доля наличия договора']
                data = data.drop(columns=columns_to_drop)
                # Колонки, которые не нужно масштабировать
                exclude_columns = ['Направление', 'Год поступления', 'Количество поступивших']
                # Оставшиеся колонки для масштабирования
                columns_to_scale = [col for col in data.columns if col not in exclude_columns]
                # Применение RobustScaler к columns_to_scale
                scaler = RobustScaler()
                data[columns_to_scale] = scaler.fit_transform(data[columns_to_scale])
                # Сохранение данных в сессию
                st.session_state.data = data

  # Вкладка 2: Обучение модели
  with tab2:
      st.header('Обучение модели')
      if st.session_state.data is None:
          st.warning("Сначала загрузите данные на вкладке 'Загрузка данных'")
      else:
          if st.button('Начать обучение'):
              with st.spinner('Обучение модели...'):
                  try:
                      # Импорт необходимых библиотек для модели
                      import tensorflow as tf
                      from keras.models import Sequential
                      from keras.layers import Dense, Input, LSTM, Dropout
                      from sklearn.model_selection import train_test_split
                      from sklearn.preprocessing import RobustScaler
                      from keras.callbacks import EarlyStopping

                      X = st.session_state.data.drop(columns=['Количество поступивших'])
                      y = st.session_state.data['Количество поступивших']
                      # Функция для создания последовательностей данных
                      def create_sequences(data, window_size):
                          inputs, outputs, groups = [], [], []
                          # Обрабаботка каждого направления отдельно
                          for direction in data['Направление'].unique():
                              # Фильтрация данных по направлению и сортировка по году
                              dir_data = data[data['Направление'] == direction].sort_values('Год поступления')
                              # Создание скользящего окна для текущего направления
                              for i in range(len(dir_data) - window_size):
                                  seq = dir_data.iloc[i:i+window_size].drop(['Направление', 'Год поступления', 'Количество поступивших'], axis=1).values
                                  target = dir_data.iloc[i+window_size]['Количество поступивших']
                                  direction_label = dir_data.iloc[i+window_size]['Направление']  # Направление для следующего шага
                                  # Добавление результатов в списки
                                  inputs.append(seq)
                                  outputs.append(target)
                                  groups.append(direction_label)
                          return np.array(inputs), np.array(outputs), np.array(groups)

                      # Создание последовательностей размером 5 лет
                      YEARS_SIZE = 5
                      X, y, direction_labels = create_sequences(st.session_state.data, YEARS_SIZE)

                      # Нормализация целевой переменной
                      y_scaler = RobustScaler()
                      y = y_scaler.fit_transform(y.reshape(-1, 1)).flatten()

                      # Разделение данных на тренировочную и тестовую выборки
                      X_train, X_test, y_train, y_test, direction_train, direction_test = train_test_split(
                          X, y, direction_labels, test_size=0.2, random_state=42)

                      # Сохрание x_test и y_test в сессию
                      st.session_state.X_test = X_test
                      st.session_state.y_test = y_test

                      # Рассчитываются веса только для тренировочной выборки
                      unique_directions, direction_counts_train = np.unique(direction_train, return_counts=True)
                      direction_weights_train = {direction: 1.0 / count for direction, count in zip(unique_directions, direction_counts_train)}

                      # Применение весов к данным
                      sample_weights = st.session_state.data['Направление'].map(direction_weights_train).values

                      # Ограничение веса для тренировочной выборки
                      train_sample_weights = sample_weights[:len(y_train)]

                      # Архитектура модели
                      model_lstm = Sequential([
                          Input(shape=(YEARS_SIZE, X.shape[2])), # Входной слой, размер окна 5, количество признаков - 2
                          LSTM(128, return_sequences=True), # Первый lstm-слой из 64 нейронов
                          Dropout(0.1), # Слой для предовращения обучения 10% отключение нейронов
                          LSTM(64), # Второй lstm-слой из 64 нейронов
                          Dense(32, activation='relu'), # Полносвязный слой из 32 нейронов, сжимает признаки
                          Dense(1) # Выходной слой предсказывающий количество абитуриентов
                      ])

                      # Компиляция модели с оптимизатором adam
                      model_lstm.compile(optimizer='adam', loss='mae', metrics=['mae'])

                      # Ранняя остановка при переобучении с параметром 10
                      early_stopping = EarlyStopping(
                          monitor='val_loss',
                          patience=10,
                          restore_best_weights=True  # Восстановление весов модели с лучшим результатом
                      )

                      # Обучение модели и сохранение в переменнную для отображения истории обучения
                      history_lstm = model_lstm.fit(
                          X_train,
                          y_train,
                          sample_weight=train_sample_weights,  # Использование веса
                          epochs=100, # Количество эпох 10
                          batch_size=64, # Размер батча 64
                          validation_split=0.2,
                          callbacks=[early_stopping],
                          verbose=1
                      )

                      # Сохранение модели в сессию
                      model_lstm.save('lstm_model.keras')
                      st.success('Модель успешно обучена и сохранена!')
                      st.session_state.model = model_lstm

                      # Сохранение scaler для предсказаний
                      st.session_state.scaler = y_scaler

                      # Визуализация обучения
                      st.line_chart(pd.DataFrame(history_lstm.history))

                      # Возможность скачивания файла в формате .keras
                      with open('lstm_model.keras', 'rb') as file:
                        st.download_button(
                            label='Скачать модель',
                            data=file,
                            file_name='lstm_model.keras',
                            mime='application/octet-stream'
                            )
                  except Exception as e:
                    st.error(f'Ошибка при обучении модели: {e}')

  # Вкладка 3: Тестирование модели
  with tab3:
      st.header('Тестирование модели')
      # Загрузка модели в формате .keras
      st.subheader('Загрузка модели')
      uploaded_model = st.file_uploader('Выберите файл модели (.keras)', type=['keras'])
      if uploaded_model:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.keras') as tmp_file:
          tmp_file.write(uploaded_model.getvalue())
          st.session_state.model = load_model(tmp_file.name)
        st.success('Модель успешно загружена!')

      if st.session_state.model is None:
          st.warning('Сначала загрузите или обучите модель')
      else:
          # Подраздел: Оценка модели
          st.subheader('Оценка модели')
          if st.button('Оценить модель'):
              try:
                  # Извлечение тестовых данных из сессии
                  X_test = st.session_state.X_test
                  y_test = st.session_state.y_test

                  # Предсказания на тестовых данных
                  predictions = st.session_state.model.predict(X_test)
                  predictions = st.session_state.scaler.inverse_transform(predictions)
                  y_test_original = st.session_state.scaler.inverse_transform(y_test.reshape(-1, 1))

                  # Оценка качества модели MAE и R²
                  from sklearn.metrics import mean_absolute_error, r2_score
                  mae = mean_absolute_error(y_test_original, predictions)
                  r2 = r2_score(y_test_original, predictions)
                  st.write(f'Средняя абсолютная ошибка (MAE): {mae:.2f}')
                  st.write(f'Коэффициент детерминации (R²): {r2:.2f}')

                  # Отображение результатов в виде таблицы
                  results = pd.DataFrame({
                      'Реальные значения': y_test_original.flatten(),
                      'Предсказанные значения': predictions.flatten()
                  })
                  st.write('Результаты предсказания на тестовых данных:')
                  st.dataframe(results)

              except Exception as e:
                  st.error(f'Ошибка при оценке модели: {e}')
          # Подраздел: Прогнозирование
          st.subheader('Прогнозирование')
          all_directions = st.session_state.data['Направление'].unique().tolist()
          forecast_option = st.selectbox('Выберите направления для прогнозирования:', ['Все'] + all_directions)
          selected_directions = all_directions if forecast_option == "Все" else [forecast_option]
          if st.button('Запустить прогнозирование'):
              try:
                  # Функция для предсказания следующего года
                  def predict_next_year(model, df, years_size, y_scaler):
                      predictions = []
                      # Выбор либо всех направлений, либо одного
                      for direction in df['Направление'].unique():
                          if selected_directions != 'Все' and direction not in selected_directions:
                              continue

                          dir_data = df[df['Направление'] == direction].sort_values('Год поступления')

                          # Если данных недостаточно для создания последовательности, пропуск направления
                          if len(dir_data) < years_size:
                              st.write(f'Недостаточно данных для направления {direction}. Требуется минимум {years_size} года.')
                              continue

                          # Создание последовательности из последних years_size лет
                          last_sequence = dir_data.iloc[-years_size:].drop(['Направление', 'Год поступления', 'Количество поступивших'], axis=1).values
                          last_sequence = last_sequence.reshape(1, years_size, last_sequence.shape[1])

                          # Нормализация последовательности
                          scaler = RobustScaler()
                          last_sequence = scaler.fit_transform(last_sequence.reshape(-1, last_sequence.shape[2])).reshape(last_sequence.shape)

                          # Предсказание количества абитуриентов
                          predicted_value = model.predict(last_sequence)

                          # Преобразование предсказания обратно в исходный масштаб
                          predicted_value = y_scaler.inverse_transform(predicted_value)[0][0]

                          # Сохранение результата
                          predictions.append({
                              'Направление': direction,
                              'Предсказанное количество абитуриентов': int(round(predicted_value))
                          })

                      # Создание DataFrame с результатами предсказания
                      result_df = pd.DataFrame(predictions)
                      return result_df

                  # Вызов функции предсказания
                  YEARS_SIZE = 5
                  predictions_df = predict_next_year(st.session_state.model, st.session_state.data, YEARS_SIZE, st.session_state.scaler)

                  if predictions_df.empty:
                      st.error('Нет данных для выбранных направлений или недостаточно записей для прогнозирования.')
                  else:
                      # Отображение результатов на следующий учебный год
                      st.write('Прогноз количества абитуриентов на следующий год:')
                      st.dataframe(predictions_df)

                      # Сохранение результатов прогноза в сессию
                      predictions_df.to_csv('results.csv', index=False)
                      st.success('Результаты прогноза успешно сохранены в файл results.csv!')
                      # Возможность скачивания файла формата CSV
                      with open('results.csv', 'rb') as file:
                        st.download_button(
                            label='Скачать результаты прогноза',
                            data=file,
                            file_name='results.csv',
                            mime='text/csv'
                            )
              except Exception as e:
                  st.error(f'Ошибка при прогнозировании: {e}')