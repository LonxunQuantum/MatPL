import unittest
import warnings

from src.pre_data.nep_data_loader import UniDataset


class EmptyNepDatasetTests(unittest.TestCase):
    def test_empty_dataset_has_zero_average_atoms_without_runtime_warning(self):
        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            dataset = UniDataset(
                [],
                "pwmlff/npy",
                [3, 14, 6],
                cutoff_radial=8.0,
                cutoff_angular=4.0,
                cal_energy=False,
            )

        self.assertEqual(len(dataset), 0)
        self.assertEqual(dataset.avg_image_atom, 0.0)


if __name__ == "__main__":
    unittest.main()
