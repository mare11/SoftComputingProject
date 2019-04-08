import tensorflow as tf
from keras.layers import Dense, Dropout, Flatten
from keras.models import Sequential


def create_network():
    # Load data
    mnist = tf.keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    # Normalizing the RGB codes by dividing it to the max RGB value.
    x_train, x_test = x_train / 255.0, x_test / 255.0

    # Creating a Sequential Model and adding the layers
    model = Sequential([
        Flatten(input_shape=(28, 28)),
        Dense(512, activation=tf.nn.relu),
        Dropout(0.2),
        Dense(10, activation=tf.nn.softmax)
    ])
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    model.fit(x_train, y_train, epochs=5)
    print('Evaluating...')
    score = model.evaluate(x_test, y_test)
    print('Baseline Error: %.2f%%' % (100 - score[1] * 100))
    print('Accuracy: %.2f%%' % (score[1] * 100))

    return model


model = create_network()
# model = None
