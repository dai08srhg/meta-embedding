import pandas as pd

a = pd.DataFrame({
    'target_id_1': [1, 2, 3, 4, 5, 2, 3, 4],
    'target_id_2': [3, 1, 2, 4, 1, 2, 3, 4],
    'meta_feature_1': ['aa', 'bb', 'cc', 'dd', 'ee', 'bb', 'cc', 'dd'],
    'meta_feature_2': [10, 20, 30, 40, 50, 20, 30, 40],
    'meta_feature_3': [5, 10, 15, 20, 25, 10, 15, 20],
    'feature_1': [1, 1, 8, 5, 5, 10, 11, 14],
    'feature_2': [11, 32, 8, 19, 5, 11, 16, 40],
    'feature_3': [30, 22, 19, 10, 20, 1, 4, 20],
    'click': [1, 0, 1, 1, 0, 0, 1, 0]
})

print(a)
a.to_pickle('/workspace/data/sample.pkl')

df_test = pd.DataFrame({
    'target_id_1': [1, 2, 100, 5],
    'target_id_2': [3, 1, 2, 100],
    'meta_feature_1': ['aa', 'bb', 'dd', 'ee'],
    'meta_feature_2': [10, 20, 30, 20],
    'meta_feature_3': [5, 10, 10, 20],
    'feature_1': [1, 1, 5, 14],
    'feature_2': [11, 32, 40, 11],
    'feature_3': [30, 22, 1, 4],
    'click': [1, 0, 0, 0]
})
df_test.to_pickle('/workspace/data/sample_test.pkl')
