
class ObjectDetectionNet:
    def __init__(self):
        pass

    def fit_generator(self, generator, validation_data):
        return self._model.fit_generator(
            generator=generator,
            validation_data=validation_data
        )

    def evaluate_generator(self, generator):
        return self._model.evaluate_generator(generator)

    def predict_generator(self, generator):
        return self._model.predict_generator(generator)
