import tempfile
import unittest
from pathlib import Path

from src.pre_data.nep_lmdb_dataset import discover_aselmdb_files
from src.user.input_param import InputParam
from src.user.work_file_param import WorkFileStructure


def _minimal_nep_json(lmdb_path, **overrides):
    config = {
        "model_type": "NEP",
        "atom_type": [1],
        "model": {
            "descriptor": {
                "cutoff": [6.0, 4.0],
                "n_max": [4, 4],
                "basis_size": [8, 8],
                "l_max": [4, 2, 1],
            },
            "fitting_net": {"network_size": [8, 1]},
        },
        "optimizer": {"optimizer": "ADAM", "epochs": 1, "batch_size": 1},
        "format": "lmdb",
        "train_data": [str(lmdb_path)],
    }
    config.update(overrides)
    return config


class DiscoverAseLmdbFilesTest(unittest.TestCase):
    def test_recursively_discovers_sorts_and_deduplicates_files(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            first = root / "z" / "first.aselmdb"
            second = root / "a" / "nested" / "second.aselmdb"
            ignored = root / "a" / "not-lmdb.txt"
            for path in (first, second, ignored):
                path.parent.mkdir(parents=True, exist_ok=True)
                path.touch()

            result = discover_aselmdb_files([root, first, root])

            self.assertEqual(result, sorted({str(first.resolve()), str(second.resolve())}))

    def test_rejects_empty_directory(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            with self.assertRaisesRegex(ValueError, "No .aselmdb files"):
                discover_aselmdb_files([tmpdir])

    def test_rejects_non_lmdb_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "data.json"
            path.touch()
            with self.assertRaisesRegex(ValueError, "Expected an .aselmdb"):
                discover_aselmdb_files([path])

    def test_rejects_missing_path(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            missing = Path(tmpdir) / "missing"
            with self.assertRaisesRegex(ValueError, "does not exist"):
                discover_aselmdb_files([missing])


class LmdbWorkFileStructureTest(unittest.TestCase):
    def test_lmdb_paths_are_expanded_for_all_dataset_roles(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            train = root / "train" / "a.aselmdb"
            valid = root / "valid" / "b.aselmdb"
            test = root / "test" / "c.aselmdb"
            for path in (train, valid, test):
                path.parent.mkdir(parents=True)
                path.touch()
            paths = WorkFileStructure(
                json_dir=tmpdir,
                reserve_work_dir=False,
                reserve_feature=False,
                model_type="NEP",
            )

            paths.set_train_valid_file(
                {
                    "model_type": "NEP",
                    "format": "lmdb",
                    "train_data": [train.parent],
                    "valid_data": valid,
                    "test_data": [test],
                }
            )

            self.assertEqual(paths.train_data_path, [str(train.resolve())])
            self.assertEqual(paths.valid_data_path, [str(valid.resolve())])
            self.assertEqual(paths.test_data_path, [str(test.resolve())])


class LmdbStatisticsConfigurationTest(unittest.TestCase):
    def test_default_and_explicit_statistics_frame_counts(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            lmdb_path = Path(tmpdir) / "tiny.aselmdb"
            lmdb_path.touch()

            default_param = InputParam(_minimal_nep_json(lmdb_path), "TRAIN")
            explicit_param = InputParam(
                _minimal_nep_json(lmdb_path, lmdb_stat_frames=8192), "TRAIN"
            )

            self.assertEqual(default_param.lmdb_stat_frames, 32768)
            self.assertEqual(explicit_param.lmdb_stat_frames, 8192)
            self.assertEqual(explicit_param.to_dict()["lmdb_stat_frames"], 8192)

    def test_statistics_frame_count_must_be_a_positive_integer(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            lmdb_path = Path(tmpdir) / "tiny.aselmdb"
            lmdb_path.touch()
            for invalid in (0, -1, True, 1.5, "4096"):
                with self.subTest(invalid=invalid):
                    with self.assertRaisesRegex(ValueError, "lmdb_stat_frames"):
                        InputParam(
                            _minimal_nep_json(
                                lmdb_path, lmdb_stat_frames=invalid
                            ),
                            "TRAIN",
                        )


if __name__ == "__main__":
    unittest.main()
