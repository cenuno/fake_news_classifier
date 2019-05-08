from sklearn.metrics import accuracy_score

def evaluate_model(predict_fun, X_train, y_train, X_test, y_test):
    """evaluate the model, both training and testing errors are reported"""
    # training error
    y_predict_train = predict_fun(X_train)
    train_acc = accuracy_score(y_train,y_predict_train)
    
    # testing error
    y_predict_test = predict_fun(X_test)
    test_acc = accuracy_score(y_test,y_predict_test)
    
    return train_acc, test_acc

from math import sqrt

def estimate_error_95_ci(error, n):
    """estimate 95% confidence interval on error."""
    term = 1.96 * sqrt((error * (1 - error)) / n)
    lb = error - term
    ub = error + term
    
    return lb, ub
