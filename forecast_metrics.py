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


def result_is_usable(result):
    """Reject missing, non-finite or misaligned stored evaluation results."""
    if not isinstance(result, dict):
        return False
    try:
        if any(not math.isfinite(float(result[key])) or float(result[key]) < 0
               for key in ("mae", "rmse", "mape")):
            return False
        actual, forecast = result["actual"], result["forecast"]
        return (len(actual) > 0 and len(actual) == len(forecast)
                and actual.index.equals(forecast.index)
                and all(math.isfinite(float(x)) for x in actual)
                and all(math.isfinite(float(x)) for x in forecast))
    except (KeyError, TypeError, ValueError, AttributeError, OverflowError):
        return False
