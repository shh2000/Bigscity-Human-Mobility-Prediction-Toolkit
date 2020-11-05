from evaluate import eval_next_loc as enl

if __name__ == '__main__':
    data = '{' \
           '"uid1": { ' \
           '"trace_id1":' \
           '{ "loc_true": [1], "loc_pred": [[0.01, 0.91, 0.8]] }, ' \
           '"trace_id2":' \
           '{ "loc_true": [2], "loc_pred": [[0.2, 0.13, 0.08]] } ' \
           '},' \
           '"uid2": { ' \
           '"trace_id1":' \
           '{ "loc_true": [0], "loc_pred": [[0.4, 0.5, 0.7]] }' \
           '}' \
           '}'
    data2 = '{' \
            '"uid1": { ' \
            '"trace_id1":' \
            '{ "loc_true": [1], "loc_pred": [[0.01, 0.1, 0.8]] }, ' \
            '"trace_id2":' \
            '{ "loc_true": [2], "loc_pred": [[0.2, 0.13, 0.08]] } ' \
            '},' \
            '"uid2": { ' \
            '"trace_id1":' \
            '{ "loc_true": [0], "loc_pred": [[0.4, 0.5, 0.7]] }' \
            '}' \
            '}'
    config = {
        'model': 'DeepMove',
        'mode': ['ACC', 'top-2', 'top-3']
    }
    var = enl.EvaluateNextLoc(config)
    # 正常写法
    #var.evaluate(data=data)
    # iterator/yield 写法
    data_list = [data, data2]
    for data in data_list:
        var.evaluate(data)
    var.save_result(r'D:\Users\12908\Documents\git\Bigscity-Human-Mobility-Prediction-Toolkit/cache/evaluate_cache')
