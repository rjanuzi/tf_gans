from dataset.local import make_train_generator, make_validation_generator, make_test_generator


def setup_model():
    raise Exception('Not implemented yet')

def train(model):
    for imgs, tagets in make_train_generator():
        pass
        # TODO - Train with the data in RAM
    
    raise Exception('Not implemented yet')

def test(model):
    raise Exception('Not implemented yet')

def save_model(model):
    raise Exception('Not implemented yet')