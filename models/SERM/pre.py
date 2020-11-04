import config
import time
import pickle
import numpy as np
import os

TWEET_PATH = config.TWEET_PATH
POI_PATH = config.POI_PATH
GRID_COUNT = config.GRID_COUNT
BATCH_SIZE = config.batch_size
WINDOW_SIZE = config.window_size
MIN_SEQ = config.min_seq_num
MAX_SEQ = config.max_seq_num
MIN_TRAJ = config.min_traj_num
RECORD_TH = config.threshold
TRAIN_TEST_PART = config.train_test_part


def time_diff(time1, time2, form='%Y-%m-%d %X'):
    time11 = time.strptime(time1, form)
    time22 = time.strptime(time2, form)
    return abs(int(time.mktime(time11)) - int(time.mktime(time22)))


def time_hour(ci_time, form='%Y-%m-%d %X'):
    st = time.strptime(ci_time, form)
    weekday = st.tm_wday
    hour = st.tm_hour
    if weekday < 6:
        return hour
    else:
        return 24 + hour


def geo_grade(index, x, y, m_nGridCount=GRID_COUNT):
    dXMax, dXMin, dYMax, dYMin = max(x), min(x), max(y), min(y)
    print(dXMax, dXMin, dYMax, dYMin)
    m_dOriginX = dXMin
    m_dOriginY = dYMin
    dSizeX = (dXMax - dXMin) / m_nGridCount
    dSizeY = (dYMax - dYMin) / m_nGridCount
    m_vIndexCells = []
    center_location_list_geo_grade = []
    for i in range(0, m_nGridCount * m_nGridCount + 1):
        m_vIndexCells.append([])
        y_ind = int(i / m_nGridCount)
        x_ind = i - y_ind * m_nGridCount
        center_location_list_geo_grade.append(
            (dXMin + x_ind * dSizeX + 0.5 * dSizeX, dYMin + y_ind * dSizeY + 0.5 * dSizeY))
    print((m_nGridCount, m_dOriginX, m_dOriginY, dSizeX, dSizeY, len(m_vIndexCells), len(index)))
    poi_index_dict = {}
    for i in range(len(x)):
        nXCol = int((x[i] - m_dOriginX) / dSizeX)
        nYCol = int((y[i] - m_dOriginY) / dSizeY)
        if nXCol >= m_nGridCount:
            print('max X')
            nXCol = m_nGridCount - 1

        if nYCol >= m_nGridCount:
            print('max Y')
            nYCol = m_nGridCount - 1

        iIndex = nYCol * m_nGridCount + nXCol
        poi_index_dict[index[i]] = iIndex
        m_vIndexCells[iIndex].append([index[i], x[i], y[i]])

    return poi_index_dict, center_location_list_geo_grade


def load_wordvec():
    word_vec = {'0'}
    return word_vec


def text_feature_generation(user_feature_sequence_gen):
    text_vec = load_wordvec()
    useful_vec_gen = {}
    print(("useful data length", len(user_feature_sequence_gen)))
    for u in list(user_feature_sequence_gen.keys()):
        features = user_feature_sequence_gen[u]
        for traj_fea in range(len(features)):
            useful_word_sample = []
            for i in range(len(features[traj_fea][2])):
                text = features[traj_fea][2][i]
                words_key = []
                if not text == 0:
                    words = text.split(' ')
                    for w in words:
                        if (w in text_vec) & (w not in useful_vec_gen):
                            useful_vec_gen[w] = text_vec[w]
                        if w in useful_vec_gen:
                            words_key.append(w)
                else:
                    print("Text == 0")
                useful_word_sample.append(words_key)
            user_feature_sequence_gen[u][traj_fea].append(useful_word_sample)
    return user_feature_sequence_gen, useful_vec_gen


def decode_data_fs(threshold=RECORD_TH):
    tsf = open(TWEET_PATH, encoding='utf8')
    poif = open(POI_PATH, encoding='utf8')
    pois = {}
    index = []
    x = []
    y = []
    for line in poif:
        poifs = line.split(',')
        if len(poifs) > 5:
            print('error')
        pois[poifs[0]] = poifs

    useful_poi = {}
    useful_user_cis = {}
    user_cis = {}
    poi_cis = {}
    poi_catecology_dict = {}
    tsfls = tsf.readlines()
    for line in tsfls:
        cifs = line.replace('\n', '').split('')
        if cifs[8] in pois:
            if cifs[8] in poi_cis:
                poi_cis[cifs[8]].append(cifs)
            else:
                poi_cis[cifs[8]] = []
                poi_cis[cifs[8]].append(cifs)

            if cifs[1] in user_cis:
                user_cis[cifs[1]].append(cifs)
            else:
                user_cis[cifs[1]] = []
                user_cis[cifs[1]].append(cifs)

            if pois[cifs[8]][3] in poi_catecology_dict:
                poi_catecology_dict[pois[cifs[8]][3]].append(pois[cifs[8]])
            else:
                poi_catecology_dict[pois[cifs[8]][3]] = []
                poi_catecology_dict[pois[cifs[8]][3]].append(pois[cifs[8]])

    for u in list(user_cis.keys()):
        if len(user_cis[u]) >= threshold:
            useful_user_cis[u] = user_cis[u]
            for r in user_cis[u]:
                if r[8] not in useful_poi:
                    useful_poi[r[8]] = pois[r[8]]
    for p in list(useful_poi.keys()):
        poifs = pois[p]
        x.append(float(poifs[1]))
        y.append(float(poifs[2]))
        index.append(poifs[0])

    print(('POI nums', len(list(useful_poi.keys()))))
    print(('User nums', len(list(useful_user_cis.keys()))))

    return useful_poi, useful_user_cis, poi_catecology_dict


def geo_data_clean_fs(w=WINDOW_SIZE, min_seq_num=MIN_SEQ, min_traj_num=MIN_TRAJ, locationtpye='GRADE',
                      gridc=GRID_COUNT):
    poi_attr, user_ci, poi_catecology_dict = decode_data_fs()
    users = list(user_ci.keys())
    user_record_sequence = {}
    useful_poi_dict = {}
    user_feature_sequence_fs = {}

    # use W and min_traj_num filter data
    for user in users:
        ci_records = user_ci[user]
        ci_records.reverse()
        clean_records = []
        traj_records = []
        perious_record = None
        for record in ci_records:
            try:
                if perious_record is None:
                    perious_record = record

                time_fs = record[4]
                if time_diff(time_fs, perious_record[4]) < w:
                    traj_records.append(record)
                else:
                    if len(traj_records) > min_seq_num:
                        clean_records.append(traj_records)
                    traj_records = []
                perious_record = record
            except Exception as e:
                print(e)
        if (len(traj_records) > 0) & (len(traj_records) > min_seq_num):
            clean_records.append(traj_records)

        if len(clean_records) > min_traj_num:
            user_record_sequence[user] = clean_records

    # generate useful pois
    for user in list(user_record_sequence.keys()):
        trajs = user_record_sequence[user]
        for traj in trajs:
            for record in traj:
                if record[8] not in useful_poi_dict:
                    useful_poi_dict[record[8]] = []
                    useful_poi_dict[record[8]].append(record)

    # generate poi dict
    if locationtpye == 'GRADE':
        index, x, y = [], [], []
        for i in list(useful_poi_dict.keys()):
            poifs = poi_attr[i]
            index.append(i)
            x.append(float(poifs[1]))
            y.append(float(poifs[2]))
        poi_index_dict, center_location_list_fs = geo_grade(index, x, y, m_nGridCount=gridc)
    elif locationtpye == 'LOCS':
        poi_index_dict = {}
        locs = list(useful_poi_dict.keys())
        for p in range(len(locs)):
            poifs = locs[p]
            poi_index_dict[poifs] = p
        center_location_list_fs = []
    else:
        poi_index_dict = {}
        center_location_list_fs = []

    print(("POI Dim", len(list(poi_index_dict.keys()))))
    seg_max_record_fs = 0

    for user in list(user_record_sequence.keys()):
        all_sequ_features = []
        for traj in user_record_sequence[user]:
            pl_features = []
            time_features = []
            text_features = []
            if seg_max_record_fs < len(traj):
                seg_max_record_fs = len(traj)
            for record in traj:
                pl_features.append(poi_index_dict[record[8]] + 1)
                time_features.append(time_hour(record[4]) + 1)
                text_features.append(record[6])
            all_sequ_features.append([pl_features, time_features, text_features])
        user_feature_sequence_fs[user] = all_sequ_features
    print('seg_max_record, pois_num, user_num')
    print(seg_max_record_fs, len(list(poi_index_dict.keys())), len(list(user_feature_sequence_fs.keys())))

    user_feature_sequence_text, useful_vec_fs = text_feature_generation(user_feature_sequence_fs)

    pickle.dump((user_feature_sequence_text, poi_index_dict, seg_max_record_fs, center_location_list_fs, useful_vec_fs),
                open('./features/features&index_seg_gride_fs', 'wb'))

    return user_feature_sequence_text, poi_index_dict, seg_max_record_fs, center_location_list_fs, useful_vec_fs


def to_categorical(y, num_classes=None, dtype='float32'):
    """Converts a class vector (integers) to binary class matrix.

    E.g. for use with categorical_crossentropy.

    # Arguments
        y: class vector to be converted into a matrix
            (integers from 0 to num_classes).
        num_classes: total number of classes.
        dtype: The data type expected by the input, as a string
            (`float32`, `float64`, `int32`...)

    # Returns
        A binary matrix representation of the input. The classes axis
        is placed last.
    """
    y = np.array(y, dtype='int')
    input_shape = y.shape
    if input_shape and input_shape[-1] == 1 and len(input_shape) > 1:
        input_shape = tuple(input_shape[:-1])
    y = y.ravel()
    if not num_classes:
        num_classes = np.max(y) + 1
    n = y.shape[0]
    categorical = np.zeros((n, num_classes), dtype=dtype)
    categorical[np.arange(n), y] = 1
    output_shape = input_shape + (num_classes,)
    categorical = np.reshape(categorical, output_shape)
    return categorical


def text_features_to_categorical(text_features_train, word_index):
    textf_res = []
    for item in text_features_train:
        if item == 0:
            textf_res.append(np.zeros(len(list(word_index.keys()))))
        elif len(item) == 0:
            textf_res.append(np.zeros(len(list(word_index.keys()))))
        else:
            line = len(item)
            vec = np.zeros(len(list(word_index.keys())))
            for w in item:
                wv = to_categorical([word_index[w]], len(list(word_index.keys())))
                vec = vec + wv
            vec = vec / line
            textf_res.append(vec[0])
    return textf_res


def geo_dataset_train_test_text(user_feature_sequence_divide, useful_vec_divide, max_record,
                                place_dim=GRID_COUNT * GRID_COUNT, train_test_part=TRAIN_TEST_PART):
    user_index = {}
    for u in range(len(list(user_feature_sequence_divide.keys()))):
        user_index[list(user_feature_sequence_divide.keys())[u]] = u
    user_dim = len(list(user_feature_sequence_divide.keys()))

    word_index = {}
    word_vec = []
    for w in range(len(list(useful_vec_divide.keys()))):
        word_index[list(useful_vec_divide.keys())[w]] = w
        word_vec.append(useful_vec_divide[list(useful_vec_divide.keys())[w]])
    word_vec = np.array(word_vec)
    print(word_vec.shape)
    all_train_X_pl, all_train_X_time, all_train_X_user, all_train_X_text, all_train_Y, all_train_evl = \
        [], [], [], [], [], []
    all_test_X_pl, all_test_X_time, all_test_X_user, all_test_X_text, all_test_Y, all_test_evl = \
        [], [], [], [], [], []

    for user in list(user_feature_sequence_divide.keys()):
        sequ_features = user_feature_sequence_divide[user]
        train_size = int(len(sequ_features) * train_test_part) + 1
        for sample in range(0, train_size):
            pl_features, time_features, text_features_train = \
                sequ_features[sample][0], sequ_features[sample][1], sequ_features[sample][3]
            pl_train = pl_features[0:len(pl_features) - 1]
            time_train = time_features[0:len(time_features) - 1]
            user_index_train = [(user_index[user] + 1) for item in range(len(pl_features) - 1)]
            text_features_train = text_features_train[0:len(text_features_train) - 1]
            while len(pl_train) < (max_record - 1):
                pl_train.append(0)
                time_train.append(0)
                user_index_train.append(0)
                text_features_train.append(0)
            train_y = pl_features[1:]
            while len(train_y) < (max_record - 1):
                train_y.append(0)
            all_train_X_pl.append(np.array(pl_train))
            all_train_X_time.append(np.array(time_train))
            all_train_X_user.append(np.array(user_index_train))
            all_train_X_text.append(text_features_train)
            all_train_Y.append(train_y)
            all_train_evl.append(train_y)

        for sample in range(train_size, len(sequ_features)):
            pl_features, time_features, text_features_test = \
                sequ_features[sample][0], sequ_features[sample][1], sequ_features[sample][3]
            pl_test = pl_features[0:len(pl_features) - 1]
            time_test = time_features[0:len(time_features) - 1]
            user_index_test = [(user_index[user] + 1) for item in range(len(pl_features) - 1)]
            text_features_test = text_features_test[0:len(text_features_test) - 1]
            while len(pl_test) < (max_record - 1):
                pl_test.append(0)
                time_test.append(0)
                user_index_test.append(0)
                text_features_test.append(0)
            test_y = pl_features[1:]
            while len(test_y) < (max_record - 1):
                test_y.append(0)
            all_test_X_pl.append(np.array(pl_test))
            all_test_X_time.append(np.array(time_test))
            all_test_X_user.append(np.array(user_index_test))
            all_test_X_text.append(text_features_to_categorical(text_features_test, word_index))
            all_test_Y.append(to_categorical(test_y, num_classes=place_dim + 1))
            all_test_evl.append(test_y)

    print(all_train_X_pl[0])
    print(all_train_evl[0])
    all_train_X_pl = np.array(all_train_X_pl)
    all_train_X_time = np.array(all_train_X_time)
    all_train_X_user = np.array(all_train_X_user)
    all_train_evl = np.array(all_train_evl)
    all_train_Y = np.array(all_train_Y)
    all_test_X_pl = np.array(all_test_X_pl)
    all_test_X_time = np.array(all_test_X_time)
    all_test_X_user = np.array(all_test_X_user)
    all_test_X_text = np.array(all_test_X_text)

    print(("all_train_X_pl,all_train_X_time,all_train_X_user",
           all_train_X_pl.shape, all_train_X_time.shape, all_train_X_user.shape))
    return [all_train_X_pl, all_train_X_time, all_train_X_user, all_train_X_text], np.array(
        all_train_Y), all_train_evl, [all_test_X_pl, all_test_X_time, all_test_X_user, all_test_X_text], np.array(
        all_test_Y), all_test_evl, user_dim, word_vec, word_index


def text_features_to_categorical_batch(text_features_train_batch, word_index):
    textf_res_batch = []
    for text_features_train in text_features_train_batch:
        textf_res = text_features_to_categorical(text_features_train, word_index)
        textf_res_batch.append(textf_res)
    return textf_res_batch


def pre():
    feature_path = './features/features&index_seg_gride_fs'
    if os.path.exists(feature_path):
        user_feature_sequence, place_index, seg_max_record, center_location_list, useful_vec = pickle.load(
            open(feature_path, 'rb'))
    else:
        user_feature_sequence, place_index, seg_max_record, center_location_list, useful_vec = geo_data_clean_fs()
    print(place_index)
    print(len(list(user_feature_sequence.keys())))
    train_X, train_Y, train_evl, vali_X, vali_Y, vali_evl, user_dim, word_vec, word_index = geo_dataset_train_test_text(
        user_feature_sequence, useful_vec, seg_max_record)
    print("Feature generation completed")
    return train_X, train_Y, train_evl, vali_X, vali_Y, vali_evl, user_dim, word_vec, word_index, center_location_list, seg_max_record


if __name__ == '__main__':
    pre()
