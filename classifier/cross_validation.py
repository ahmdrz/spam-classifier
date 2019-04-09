from sklearn.model_selection import KFold

def kfold_cross_validation(data, k=10):
    kfold = KFold(n_splits=k)
    for train, test in kfold.split(data):
        yield data[train], data[test]    