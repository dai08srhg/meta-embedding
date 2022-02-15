import pandas as pd

a = pd.DataFrame({
    'ad_id': [1, 2, 3, 4, 5, 2, 3, 4],
    'ad_feature_1': ['aa', 'bb', 'cc', 'dd', 'ee', 'bb', 'cc', 'dd'],
    'ad_feature_2': [10, 20, 30, 40, 50, 20, 30, 40],
    'ad_feature_3': [5, 10, 15, 20, 25, 10, 15, 20],
    'other_feature_1': [1, 1, 8, 5, 5, 10, 11, 14],
    'other_feature_2': [11, 32, 8, 19, 5, 11, 16, 40],
    'other_feature_3': [30, 22, 19, 10, 20, 1, 4, 20],
    'click': [1, 0, 1, 1, 0, 0, 1, 0]
})

print(a)
a.to_pickle('/workspace/data/sample.pkl')