import numpy as np

from src.data_processing.churn_preprocessor import preprocess_churn_data


def test_output_shape():
    X_train, X_test, _, _ = preprocess_churn_data(save_artifacts=False)
    assert X_train.shape[1] == 9
    assert X_test.shape[1] == 9


def test_stratified_split():
    _, _, y_train, y_test = preprocess_churn_data(save_artifacts=False)
    assert abs(float(y_train.mean()) - float(y_test.mean())) <= 0.05


def test_no_nulls():
    X_train, X_test, y_train, y_test = preprocess_churn_data(save_artifacts=False)
    assert not np.isnan(X_train).any()
    assert not np.isnan(X_test).any()
    assert not np.isnan(y_train).any()
    assert not np.isnan(y_test).any()
