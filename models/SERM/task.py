import config
from pre import pre, to_categorical, text_features_to_categorical_batch
from evaluate import evaluation_last_with_distance
from model_torch import serm_torch
import threading
import numpy as np
import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

os.environ["CUDA_VISIBLE_DEVICES"] = config.GPU

PRETRAINED_FS = config.PRETRAINED_FS
GRID_COUNT = config.GRID_COUNT
BATCH_SIZE = config.batch_size
MODEL_NAME = config.model_file_name
TRAINING_EPOCH = config.training_epoch


class threadsafe_iter:
    """Takes an iterator/generator and makes it thread-safe by
    serializing call to the `next` method of given' iterator/generator.
    """

    def __init__(self, it):
        self.it = it
        self.lock = threading.Lock()

    def __iter__(self):
        return self

    def __next__(self):
        with self.lock:
            return next(self.it)


def threadsafe_generator(f):
    """A decorator that takes a generator function and makes it thread-safe.
    """

    def g(*a, **kw):
        return threadsafe_iter(f(*a, **kw))

    return g


@threadsafe_generator
def batch_generator_text(train_X_batch, train_Y_batch, word_index_batch):
    place_dim_batch = GRID_COUNT * GRID_COUNT
    while 1:
        j = 0
        while j < train_X_batch[0].shape[0]:
            y_b = []
            pl_b, t_b, user_b = train_X_batch[0][j:j + BATCH_SIZE], train_X_batch[1][j:j + BATCH_SIZE], train_X_batch[
                                                                                                            2][
                                                                                                        j:j + BATCH_SIZE]
            text_b = np.array(text_features_to_categorical_batch(train_X_batch[3][j:j + BATCH_SIZE], word_index_batch))
            for sample in train_Y_batch[j:j + BATCH_SIZE]:
                y_b.append(to_categorical(sample, num_classes=place_dim_batch + 1))
            yield [pl_b, t_b, user_b, text_b], np.array(y_b)

            if (j + BATCH_SIZE) > train_X_batch[0].shape[0]:
                y_b = []
                pl_b, t_b, user_b = train_X_batch[0][j:], train_X_batch[1][j:], train_X_batch[2][j:]
                text_b = np.array(text_features_to_categorical_batch(train_X_batch[3][j:], word_index_batch))
                for sample in train_Y_batch[j:]:
                    y_b.append(to_categorical(sample, num_classes=place_dim_batch + 1))
                print((pl_b.shape, t_b.shape, text_b.shape, user_b.shape))
                yield [pl_b, t_b, user_b, text_b], np.array(y_b)
            j = j + BATCH_SIZE


train_X, train_Y, train_evl, vali_X, vali_Y, vali_evl, user_dim, word_vec, word_index, center_location_list, seg_max_record = pre()

model = serm_torch(user_dim, seg_max_record, word_vec)

# model.load_weights(PRETRAINED_FS)
all_output_array = model.forward(vali_X)
# evaluation_last_with_distance(all_output_array, vali_evl, center_location_list)
print(("Train_x[0] shape:", train_X[1].shape))
print(("Train_x[0] shape:", train_X[2].shape))
print(("Train_Y shape:", train_Y.shape))
quit(0)
