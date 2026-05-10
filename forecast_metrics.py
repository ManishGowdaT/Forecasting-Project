"""Consistent evaluation of point forecasts for weekly sales."""
import math


def calculate_metrics(actual, predicted):
    """Return MAE, RMSE and percentage MAPE (excluding zero actuals)."""
    a = [float(x) for x in actual]
    p = [float(x) for x in predicted]
    if not a or len(a) != len(p):
        raise ValueError("Actual and predicted series must have equal nonzero length")
    if not all(math.isfinite(x) for x in a + p):
        raise ValueError("Evaluation values must be finite")
    errors = [abs(x - y) for x, y in zip(a, p)]
    percentage = [e / abs(x) for x, e in zip(a, errors) if x != 0]
    if not percentage:
        raise ValueError("MAPE is undefined when every actual value is zero")
    return (math.fsum(errors) / len(a),
            math.sqrt(math.fsum(e * e for e in errors) / len(a)),
            100 * math.fsum(percentage) / len(percentage))
