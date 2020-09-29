from evaluate import evaluate as lpem

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
    # Eval.evaluate()
    Eval.evaluate(data=data, mode=['ACC', 'MAE', 'top-2', 'top-3'])
    Eval.save_result('/runtimeFiles/evaluate')
