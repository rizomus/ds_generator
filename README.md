dataset generator

    df - pd.Dataframe - данные для датасета (если None - то используется файл csv)

    csv - имя файла загруженного с помощью gdown ('train.csv') (если None - то используется датафрейм)

    column_index = [1,2,3,4,5,6,7,8] - индексы столбцов с фичами (0 - глубина)

    y_type = 0 - Коллекторы,    (15547, 3) 
             1 - KPEF,          (15547, 1)
             2 - KNEF,          (15547, 1)
             3 - KPEF + KNEF,   (15547, 2)

    scaler = 'sigmoid' - нормализация 'x' к [0..1] 
           = 'tanh'    - нормализация 'x' к [-1..1] 
             (KPEF и KNEF изначально в диапазоне [0..1])

    shuffle = True - батч из случайных данных, False - бачи формируются последовательно
