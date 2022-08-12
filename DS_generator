from tensorflow.keras.utils import Sequence
from sklearn.preprocessing import OneHotEncoder
import pandas as pd
import numpy as np
import gdown


class DS_generator(Sequence):

    '''
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
    '''

    def __init__(self,
                 df = None,
                 csv = None,
                 column_index = [1,2,3,4,5,6,7,8],
                 y_type = 0,                           
                 scaler = 'sigmoid',
                 shuffle = True,
                 batch_size = 64):

        if csv:
            df = pd.read_csv(csv, decimal=',')
        else:
            df = df.copy()
            
        self.column_index = column_index
        self.scler = scaler
        self.y_type = y_type
        self.shuffle = shuffle
        self.batch_size = batch_size
        self.x = df.iloc[:,column_index].values.astype(np.float32)
        
        if y_type == 0:
            enc = OneHotEncoder()
            self.y = enc.fit_transform(df['Коллекторы'].values.reshape(-1, 1)).toarray().astype(np.int16)
        
        elif y_type == 1:
            self.y = df.loc[:, 'KPEF'].values
            self.y = np.expand_dims(self.y, axis=1)

        elif y_type == 2:
            self.y = df.loc[:, 'KNEF'].values
            self.y = np.expand_dims(self.y, axis=1)

        elif y_type == 3:
            self.y = df.loc[:, 'KPEF': 'KNEF'].values
        
        self.x_max = []                       # максимальное значение в каждом столбце для нормализации
        self.x_min = []

        for i in range(self.x.shape[1]):
            self.x_max.append(self.x[:,i].max())
            self.x_min.append(self.x[:,i].min())


    def sigmoid_scaler(self, x, x_max, x_min):
        return (x - x_min) / (x_max - x_min)


    def tanh_scaler(self, x, x_max, x_min):
        return (x - x_min) / (x_max - x_min) * 2 - 1


    def add_korr_and_norm(self, x):         # внесение случайной поправки в первые пять признаков 
                                           # и нормализация x_train
        if self.scler == 'sigmoid':
            Scaler = self.sigmoid_scaler
        elif self.scler == 'tanh':
            Scaler = self.tanh_scaler

        korrs = ((-0.1, 0.1), (-1.5, 1.5), (-1, 1), (-50, 50), (-20, 20))
        korr = []

        for i in self.column_index:              # выбираем нужные поправки
            if i <= 4:                      # для первых пяти фич
                korr.append(korrs[i])

        x = list(np.transpose(x))                       # (batch, 8) -> (8, batch)

        for i in range(len(korr)):                      # для первых пяти столбцов вносим корректировки и нормализуем
            x[i] = x[i] + np.random.uniform(*korr[i])
            x[i] = Scaler(x[i], (self.x_max[i]+korr[i][1]), (self.x_min[i]-korr[i][1]))

        for i in range(len(korr), len(x)):              # для остальных только нормализуем
            x[i] = Scaler(x[i], self.x_max[i], self.x_min[i])

        x = np.transpose(x)                 # обратно в нужную размерность
        return x


    def __len__(self):                                      # возвращает количество бачей в датасете
        return len(self.x) // self.batch_size


    def __getitem__(self, idx):                                         # этот метод формирует батч
        if self.shuffle == True:
            indices = np.random.choice(len(self.x), size=self.batch_size)
        else:
            indices = np.arange(idx * self.batch_size, (idx + 1) * self.batch_size)
        
        indices = np.random.choice(len(self.x), size=self.batch_size)         # случайные индексы для бача (idx - номер бача в эпохе(?) - не используется)
        x_batch = self.x[indices]
        y_batch = self.y[indices]
        x_batch = self.add_korr_and_norm(x_batch)                 # случайная поправка для 5 фичей
        x_batch = np.expand_dims(x_batch, axis=0)
        y_batch = np.expand_dims(y_batch, axis=0)
        return x_batch, y_batch
