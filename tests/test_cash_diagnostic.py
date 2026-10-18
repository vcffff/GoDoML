from scripts.diagnose_cash_data import state
from scripts.diagnose_cash_overfit import metrics


def test_unknown_and_errors_are_never_counted_as_negatives():
    rows = [
        {'expected': 'violation', 'prediction': 'unknown'},
        {'expected': 'no_violation', 'prediction': None},
        {'expected': 'unknown', 'prediction': 'no_violation'},
        {'expected': 'violation', 'prediction': 'violation'},
        {'expected': 'no_violation', 'prediction': 'no_violation'},
        {'expected': 'no_violation', 'prediction': 'violation'},
        {'expected': 'violation', 'prediction': 'no_violation'},
    ]
    result = metrics(rows)
    assert result == {'TP': 1, 'FP': 1, 'TN': 1, 'FN': 1, 'unknown': 1,
                      'errors': 1, 'n': 7, 'correct': 2}
    assert [state(v) for v in [True, False, 'unknown']] == [
        'violation', 'no_violation', 'unknown']
