import unittest

from e1.boundary_calibration import BoundaryCalibrator, BoundaryShift
from e1.schemas import EvidenceSpan
from dev.boundary_tuning import best_shift


class BoundaryCalibrationTests(unittest.TestCase):
    def test_source_specific_shift(self):
        calibrator = BoundaryCalibrator(
            global_shift=BoundaryShift(0.0, 0.0),
            by_source={"medasr": BoundaryShift(-0.1, 0.2)},
        )
        span = EvidenceSpan("medasr", 1, 3, 2.0, 3.0)
        out = calibrator.apply(span)
        self.assertAlmostEqual(out.start, 1.9)
        self.assertAlmostEqual(out.end, 3.2)

    def test_grid_tuner_finds_expansion(self):
        records = [
            {
                "raw_start": 1.1,
                "raw_end": 1.9,
                "gold_start": 1.0,
                "gold_end": 2.0,
            }
            for _ in range(5)
        ]
        start, end, score = best_shift(records, [-0.1, 0.0, 0.1])
        self.assertEqual(start, -0.1)
        self.assertEqual(end, 0.1)
        self.assertAlmostEqual(score, 1.0)


if __name__ == "__main__":
    unittest.main()
