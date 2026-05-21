import math
import unittest
from forecast_metrics import calculate_metrics


class MetricTests(unittest.TestCase):
    def test_known_errors(self):
        mae, rmse, mape = calculate_metrics([100, 200], [90, 220])
        self.assertAlmostEqual(mae, 15)
        self.assertAlmostEqual(rmse, math.sqrt(250))
        self.assertAlmostEqual(mape, 10)

    def test_zero_actual_excluded_only_from_mape(self):
        mae, rmse, mape = calculate_metrics([0, 100], [20, 90])
        self.assertAlmostEqual(mae, 15)
        self.assertAlmostEqual(mape, 10)

    def test_undefined_mape(self):
        with self.assertRaises(ValueError):
            calculate_metrics([0, 0], [1, 2])

    def test_bad_inputs(self):
        for actual, predicted in [([], []), ([1], [1, 2]), ([1], [float("nan")])]:
            with self.subTest(actual=actual, predicted=predicted):
                with self.assertRaises(ValueError):
                    calculate_metrics(actual, predicted)


class _Index(list):
    def equals(self, other):
        return self == other


class _Series(list):
    def __init__(self, values, dates):
        super().__init__(values)
        self.index = _Index(dates)


class StoredResultTests(unittest.TestCase):
    def test_valid_and_invalid_results(self):
        from forecast_metrics import result_is_usable
        result = {"mae": 1, "rmse": 1, "mape": 1,
                  "actual": _Series([100], ["week1"]),
                  "forecast": _Series([99], ["week1"])}
        self.assertTrue(result_is_usable(result))
        self.assertFalse(result_is_usable(None))
        self.assertFalse(result_is_usable(dict(result, mape=float("inf"))))
        self.assertFalse(result_is_usable(dict(result, forecast=_Series([99], ["week2"]))))
        self.assertFalse(result_is_usable(dict(result, forecast=_Series([float("nan")], ["week1"]))))
