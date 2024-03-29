# Разработка генеративного чат-бота на примере персонажей сериала "Друзья"

**Источник данных:** https://www.kaggle.com/datasets/gopinath15/friends-netflix-script-data 

**Количество записей**: 69974

**Датасет содержит 5 полей:**

* Text – реплика персонажа и технический текст
* Speaker – персонаж, говорящий реплику
* Episode – номер и название эпизода
* Season – номер сезона
* Show – название шоу

## Примеры работы чат-бота

<img src="./valid/result_test2.png" width=auto height=auto/>

Запуск в колабе: [https://colab.research.google.com/drive/15yGyk3K_r-KSFm2ZilEsja0z9fnq8rVl](https://colab.research.google.com/drive/1IaFwWL12LMLIdNgq9u1QCkxL87_a9GAx )

## Структура репозитория
    ├── data                               # Данные
    ├────── Friends_processed.csv          # Обработанный датасет
    ├────── data.pkl                       # Данные для обучения модели
    ├── valid                              # Результаты валидации на тестах
    ├── Analysis.ipynb                     # Анализ данных
    ├── Processing.ipynb                   # Обработка данных
    ├── README.md                          # Краткое описание проекта
    ├── Report_ShmelkovYB.pdf              # Отчет
    ├── app.py                             # Веб интерфейс для huggingface space
    ├── inference.ipynb                    # Инференс
    ├── requirements.txt                   # Используемые библиотеки с версиями
    └── train.ipynb                        # Обучение модели
