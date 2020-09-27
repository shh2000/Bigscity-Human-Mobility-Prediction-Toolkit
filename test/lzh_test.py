from evaluate import basic as lpem

if __name__ == '__main__':
    Eval = lpem.Evaluate(r'D:\Users\12908\Documents\git\Bigscity-Human-Mobility-Prediction-Toolkit')
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
    Eval.evaluate(data=data, mode='ACC', topK=1)
    Eval.evaluate(data=data, mode='MAE')
    # Eval.evaluate(data=data, mode='MAPE')
    # Eval.evaluate(data=data, mode='MARE')
    Eval.evaluate(data=data, mode='MSE')
    Eval.evaluate(data=data, mode='RMSE')
    # Eval.evaluate(data=data, mode='SMAPE')
    Eval.save_result('/runtimeFiles/evaluate')
