from src import *
from data.data_loading import CocoStyleDataGenerator
from data.references import DATASETS


from keras.models import Sequential
from keras.layers import Conv2D, Flatten, Activation, Dense, Reshape
from keras.optimizers import SGD
from keras.losses import MSE
from keras.metrics import Accuracy


# libcudart.so.10.0
# libcublas.so.10.0
# libcufft.so.10.0
# libcurand.so.10.0
# libcusolver.so.10.0
# libcusparse.so.10.0
# libcudnn.so.7

if __name__ == "__main__":
    train_generator = CocoStyleDataGenerator(
        images_dir=DATASETS['coco']['train']['images'],
        labels_dir=DATASETS['coco']['train']['labels'],
        names_path=DATASETS['coco']['names']
    )
    test_generator = CocoStyleDataGenerator(
        images_dir=DATASETS['coco']['test']['images'],
        labels_dir=DATASETS['coco']['test']['labels'],
        names_path=DATASETS['coco']['names']
    )

    model = Sequential()
    model.add(Conv2D(
        filters=1024,
        kernel_size=(3, 3),
        activation='relu'
    ))
    model.add(Dense(4096))
    model.add(Dense(100 * 84))
    model.add(Reshape(target_shape=(100, 84)))
    model.add(Activation('softmax'))

    model.compile(
        optimizer=SGD(),
        loss=MSE,
        metrics=[Accuracy(), ]
    )

    model.fit_generator(
        generator=train_generator,
        validation_data=test_generator
    )
