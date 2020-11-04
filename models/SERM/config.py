# Data Parameters
GRID_COUNT = 100

# RNN Model Prarmeters
batch_size = 40
place_dim = GRID_COUNT * GRID_COUNT
time_dim = 48
pl_d = 50
time_k = 50
text_k = 50
hidden_neurons = 50
learning_rate = 0.01
model_file_name = str(batch_size) + '_' + str(pl_d) + '_' + str(hidden_neurons) + '_' + str(learning_rate)
training_epoch = 5
train_test_part = 0.8

# Decoder Parameters
threshold = 50
window_size = 36000
min_seq_num = 3
min_traj_num = 5
max_seq_num = 80

# File Paths
TWEET_PATH = './data/tweets-cikm.txt'
POI_PATH = './data/venues.txt'
PRETRAINED_FS = './pretrained/FS_trainable__200_50_50_0.0005_11.h5'
PRETRAINED_LA = './pretrained/LA_200_50_50_0.0005_7.h5'
GPU = "1"
