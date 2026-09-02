import unittest

import numpy as np
import torch

from src.pre_data.nep_lmdb_dataset import (
    LmdbEnergyStatistics,
    select_stat_indices,
)


class SelectStatIndicesTest(unittest.TestCase):
    def test_global_sample_is_deterministic_disjoint_and_requested_size(self):
        arguments = {
            "size": 100_000,
            "requested": 32_768,
            "world_size": 4,
            "seed": 2023,
        }
        first = [
            select_stat_indices(rank=rank, **arguments) for rank in range(4)
        ]
        repeated = [
            select_stat_indices(rank=rank, **arguments) for rank in range(4)
        ]

        self.assertEqual(first, repeated)
        self.assertTrue(all(local == sorted(local) for local in first))
        for left in range(4):
            for right in range(left + 1, 4):
                self.assertTrue(set(first[left]).isdisjoint(first[right]))
        self.assertEqual(len(set().union(*(set(local) for local in first))), 32_768)
        self.assertEqual([len(local) for local in first], [8192] * 4)

    def test_dataset_requested_and_per_rank_limits_are_applied(self):
        clipped_to_dataset = [
            select_stat_indices(11, 100, rank, 4, seed=9) for rank in range(4)
        ]
        clipped_to_request = [
            select_stat_indices(100, 7, rank, 4, seed=9) for rank in range(4)
        ]
        clipped_to_cap = [
            select_stat_indices(
                1000, 1000, rank, 3, seed=9, per_rank_cap=5
            )
            for rank in range(3)
        ]

        self.assertEqual(len(set().union(*(set(x) for x in clipped_to_dataset))), 11)
        self.assertEqual(len(set().union(*(set(x) for x in clipped_to_request))), 7)
        self.assertEqual(len(set().union(*(set(x) for x in clipped_to_cap))), 15)
        self.assertTrue(all(len(local) <= 5 for local in clipped_to_cap))

    def test_invalid_selection_arguments_are_rejected(self):
        invalid = [
            (10, 0, 0, 1, 1, 32768),
            (10, True, 0, 1, 1, 32768),
            (-1, 1, 0, 1, 1, 32768),
            (10, 1, 1, 1, 1, 32768),
            (10, 1, 0, 0, 1, 32768),
            (10, 1, 0, 1, 1, 0),
        ]
        for arguments in invalid:
            with self.subTest(arguments=arguments):
                with self.assertRaises(ValueError):
                    select_stat_indices(*arguments)

    def test_none_seed_remains_globally_deterministic(self):
        first = [select_stat_indices(100, 40, rank, 2, None) for rank in range(2)]
        repeated = [select_stat_indices(100, 40, rank, 2, None) for rank in range(2)]
        self.assertEqual(first, repeated)


class LmdbEnergyStatisticsTest(unittest.TestCase):
    def test_merge_matches_direct_global_least_squares_and_atom_counts(self):
        compositions = np.asarray(
            [[2, 0, 0], [0, 2, 0], [0, 0, 3], [1, 1, 1], [3, 1, 0]],
            dtype=float,
        )
        expected_shift = np.asarray([-1.25, -2.5, -4.0])
        energies = compositions @ expected_shift

        left = LmdbEnergyStatistics(3)
        right = LmdbEnergyStatistics(3)
        for composition, energy in zip(compositions[:2], energies[:2]):
            left.update(composition, energy)
        for composition, energy in zip(compositions[2:], energies[2:]):
            right.update(composition, energy)
        left.merge(right)

        direct, _, _, _ = np.linalg.lstsq(compositions, energies, rcond=None)
        np.testing.assert_allclose(left.energy_shift(), direct, rtol=1e-12, atol=1e-12)
        self.assertEqual(left.frame_count, 5)
        self.assertEqual(left.atom_count_sum, int(compositions.sum()))
        self.assertEqual(left.max_atoms, 4)
        self.assertAlmostEqual(left.average_atoms, compositions.sum() / 5)

    def test_update_from_dataset_sample_counts_type_map_without_retaining_frames(self):
        statistics = LmdbEnergyStatistics(3)
        sample = {
            "atom_type_map": torch.tensor([0, 2, 2, 1, 2]),
            "energy": torch.tensor([-14.5], dtype=torch.float64),
        }

        statistics.update_from_sample(sample)

        np.testing.assert_array_equal(statistics.ata, np.outer([1, 1, 3], [1, 1, 3]))
        np.testing.assert_array_equal(statistics.ate, np.asarray([1, 1, 3]) * -14.5)
        self.assertEqual(statistics.frame_count, 1)
        self.assertEqual(statistics.atom_count_sum, 5)
        self.assertEqual(statistics.max_atoms, 5)

    def test_empty_statistics_have_safe_atom_count_identities(self):
        statistics = LmdbEnergyStatistics(2)
        self.assertEqual(statistics.frame_count, 0)
        self.assertEqual(statistics.atom_count_sum, 0)
        self.assertEqual(statistics.max_atoms, 0)
        self.assertEqual(statistics.average_atoms, 0.0)
        with self.assertRaisesRegex(ValueError, "no frames"):
            statistics.energy_shift()

    def test_invalid_compositions_are_rejected(self):
        statistics = LmdbEnergyStatistics(2)
        for composition in ([1], [1, -1], [1, np.nan]):
            with self.subTest(composition=composition):
                with self.assertRaises(ValueError):
                    statistics.update(composition, -1.0)


if __name__ == "__main__":
    unittest.main()
