# Only CPU
# CUDA_VISIBLE_DEVICES = ""

from dataset.local import make_train_generator
from models.basic_cnn import BasicCNN

BATCH_SIZE = 500

MAX_DATASET_LOOPS = 1
NEW_WIDTH = 150
NEW_HEIGHT = 112

def train(model):
    total_imgs = 0
    loops = 0

    # The batch generator returns pairs of (imgs, target)
    for X, y in make_train_generator(target_col='is_malignant_melanoma', 
                                        batch_size=BATCH_SIZE,
                                        resize_params={'new_width': NEW_WIDTH, 'new_height': NEW_WIDTH},
                                        max_dataset_loops=MAX_DATASET_LOOPS):
        total_imgs += len(X)
        loops += 1
        
        model.fit(X, y, epochs=30)

        model.save('saved_models/basic_cnn')
        model.save('saved_models/basic_cnn.h5')

        print(total_imgs, loops)

        del X
        del y

def test(model):
    raise Exception('Not implemented yet')

def save_model(model):
    raise Exception('Not implemented yet')

train(BasicCNN((NEW_WIDTH, NEW_HEIGHT, 3), 1))
