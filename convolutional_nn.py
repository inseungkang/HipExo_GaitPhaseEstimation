from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten
from sklearn.model_selection import train_test_split

def create_model():
    # create model
    model = Sequential()
    
    # add model layers
    model.add(Conv2D(64, kernel_size=3, activation='relu'))
    model.add(Conv2D(32, kernel_size=3, activation='relu'))
    model.add(Flatten())
    model.add(Dense(4))

    return model


def load_data():
    """ return concatenated data from trials in tiral_nums"""
    data = pd.DataFrame()
    file_name = f'features/cnn_feature1.txt'
    data = data.append(pd.read_csv(file_name))
    return data


def train_model():
    model = create_model()

    # compile model using accuracy to measure model performance
    model.compile(optimizer='adam', loss='mae')

    # load data
    data = load_data()


