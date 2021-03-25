from ..monitoring.time_it import timing
from sklearn.model_selection import cross_val_score, cross_validate
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import StratifiedShuffleSplit, StratifiedKFold, train_test_split
import numpy as np
import pickle
from imblearn.under_sampling import RandomUnderSampler
from imblearn.ensemble import BalancedBaggingClassifier
from sklearn.metrics import make_scorer, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from ..logger import get_logger

logger = get_logger(__name__.split(".", 1)[-1])

@timing
def eval(X, y, config, crossvalidation, clf, random_state):

    print('Configurations: ' + str(config))

    def tp(y_true, y_pred): return confusion_matrix(y_true, y_pred, labels=[1, 0])[0, 0]
    def tn(y_true, y_pred): return confusion_matrix(y_true, y_pred, labels=[1, 0])[1, 1]
    def fp(y_true, y_pred): return confusion_matrix(y_true, y_pred, labels=[1, 0])[1, 0]
    def fn(y_true, y_pred): return confusion_matrix(y_true, y_pred, labels=[1, 0])[0, 1]

    scoring = {'accuracy': make_scorer(accuracy_score),
               'precision': make_scorer(precision_score, pos_label=1),
               'recall': make_scorer(recall_score, pos_label=1),
               'f1_score': make_scorer(f1_score, pos_label=1),
               'precision_neg': make_scorer(precision_score, pos_label=0),
               'recall_neg': make_scorer(recall_score, pos_label=0),
               'f1_score_neg': make_scorer(f1_score, pos_label=0),
               'tp': make_scorer(tp),
               'tn': make_scorer(tn),
               'fp': make_scorer(fp),
               'fn': make_scorer(fn)}


    for state in random_state:
        rus = RandomUnderSampler(random_state=state)
        X_rus, y_rus = rus.fit_resample(X, y)

        cv = StratifiedKFold(n_splits=crossvalidation, shuffle=True, random_state=42)

        scores = {}
        for element in scoring:
            scores['test_' + element] = []


        if config['oversampling_method']:
            for train_idx, test_idx, in cv.split(X_rus, y_rus):
                X_train, y_train = X_rus.iloc[train_idx], y_rus.iloc[train_idx]
                X_test, y_test = X_rus.iloc[test_idx], y_rus.iloc[test_idx]
                oversample = config['oversampling_method']
                X_train_oversampled, y_train_oversampled = oversample.fit_sample(X_train, y_train)
                clf.fit(X_train_oversampled, y_train_oversampled)
                y_pred = clf.predict(X_test)
                scores_dict = classification_report(y_test, y_pred, output_dict=True)
                scores['test_accuracy'].append(scores_dict['accuracy'])
                scores['test_precision'].append(scores_dict['1']['precision'])
                scores['test_precision_neg'].append(scores_dict['0']['precision'])
                scores['test_recall'].append(scores_dict['1']['recall'])
                scores['test_recall_neg'].append(scores_dict['0']['recall'])
                scores['test_f1_score'].append(scores_dict['1']['f1-score'])
                scores['test_f1_score_neg'].append(scores_dict['0']['f1-score'])
                scores['test_tp'].append(tp(y_test, y_pred))
                scores['test_tn'].append(tn(y_test, y_pred))
                scores['test_fp'].append(fp(y_test, y_pred))
                scores['test_fn'].append(fn(y_test, y_pred))
            for element in scores:
                scores[element] = np.array(scores[element])
        else:
            scores = cross_validate(clf.fit(X_rus, y_rus), X=X_rus, y=y_rus, cv=cv, scoring=scoring, return_estimator=False)
            print('Evaluation with crossvalidation')

        for key, value in scores.items():
            print(str(key))
            print(value)
            print('M: ' + str(value.mean()))
            print('SD: ' + str(value.std()))

        final_model = clf.fit(X,y)

    return final_model




