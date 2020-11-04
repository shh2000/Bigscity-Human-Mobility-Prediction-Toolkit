import operator
from math import radians, cos, sin, asin, sqrt


def haversine(lon1, lat1, lon2, lat2):
    """
    Calculate the great circle distance between two points
    on the earth (specified in decimal degrees)
    """
    lon1, lat1, lon2, lat2 = list(map(radians, [lon1, lat1, lon2, lat2]))

    # haversine
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    c = 2 * asin(sqrt(a))
    r = 6371
    return c * r * 1000


def evaluation_last_with_distance(all_output_array, all_test_Y, center_location_list):
    count, all_recall1, all_recall2, all_recall3, all_recall4, all_recall5, alldistance = 0., 0., 0., 0., 0., 0., 0.
    for j in range(len(all_test_Y)):
        y_test = all_test_Y[j]
        output_array = all_output_array[j]
        for i in range(len(y_test)):
            flag = False
            if (i + 1) < len(y_test):
                if (y_test[i] != 0) & (y_test[i + 1] == 0):
                    flag = True
            else:
                if y_test[i] != 0:
                    flag = True
            if flag:
                true_pl = y_test[i] - 1
                infe_pl = output_array[i]
                topd = infe_pl[1:].argsort()[-5:][::-1]
                dd = []
                for k in topd:
                    pred = center_location_list[k]
                    tr = center_location_list[true_pl]
                    d = haversine(pred[0], pred[1], tr[0], tr[1])
                    dd.append(d)
                d = min(dd)
                alldistance += d
                if true_pl in infe_pl[1:].argsort()[-1:][::-1]:
                    all_recall1 += 1
                if true_pl in infe_pl[1:].argsort()[-5:][::-1]:
                    all_recall2 += 1
                if true_pl in infe_pl[1:].argsort()[-10:][::-1]:
                    all_recall3 += 1
                if true_pl in infe_pl[1:].argsort()[-15:][::-1]:
                    all_recall4 += 1
                if true_pl in infe_pl[1:].argsort()[-20:][::-1]:
                    all_recall5 += 1
                count += 1
    print(count)
    print([all_recall1, all_recall2, all_recall3, all_recall4, all_recall5])
    print([all_recall1 / count, all_recall2 / count,
           all_recall3 / count, all_recall4 / count, all_recall5 / count, alldistance / count])
    return [all_recall1 / count, all_recall2 / count,
            all_recall3 / count, all_recall4 / count, all_recall5 / count, alldistance / count]


def nearest_location_last(vali_X, vali_evl, center_location_list):
    all_test_X_pl = vali_X[0]
    count, hc1, hc5, hc10, hc15, hc20, alldistance = 0., 0., 0., 0., 0., 0., 0.
    all_test_X_pl = all_test_X_pl.tolist()
    for j in range(len(all_test_X_pl)):
        trajl = all_test_X_pl[j]
        predict_traj = []
        for r in trajl:
            if r == 0:
                predict_traj.append(0)
            else:
                r = r - 1
                res_list = [[i, haversine(center_location_list[r][0], center_location_list[r][1],
                                          center_location_list[i][0], center_location_list[i][1])]
                            for i in range(len(center_location_list))]
                res_list.sort(key=operator.itemgetter(1))
                predict_traj.append([item[0] for item in res_list])
        ground_truth = vali_evl[j]
        for g in range(len(ground_truth)):
            flag = False
            if (g + 1) < len(ground_truth):
                if (ground_truth[g] != 0) & (ground_truth[g + 1] == 0):
                    flag = True
            else:
                if ground_truth[g] != 0:
                    flag = True
            if flag:
                ground_g = ground_truth[g] - 1
                if ground_g in predict_traj[g][0:1]:
                    hc1 += 1
                if ground_g in predict_traj[g][0:5]:
                    hc5 += 1
                if ground_g in predict_traj[g][0:10]:
                    hc10 += 1
                if ground_g in predict_traj[g][0:15]:
                    hc15 += 1
                if ground_g in predict_traj[g][0:20]:
                    hc20 += 1

                dd = []
                for k in predict_traj[g][0:5]:
                    pred = center_location_list[k]
                    tr = center_location_list[ground_g]
                    d = haversine(pred[0], pred[1], tr[0], tr[1])
                    dd.append(d)
                d = min(dd)
                # print d
                alldistance += d
                count += 1
                if count % 100 == 0:
                    print(("nearest location last", count))
    print(("nearest location last", count))
    print((hc1, hc5, hc10, hc15, hc20))
    print([hc1 / count, hc5 / count,
           hc10 / count, hc15 / count, hc20 / count, alldistance / count])
