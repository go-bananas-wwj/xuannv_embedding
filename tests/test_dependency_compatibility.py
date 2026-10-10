"""Exercise the NumPy/Arrow interface used by grid and partition artifacts."""

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


def test_numpy_columns_round_trip_through_parquet(tmp_path):
    coordinates = np.array([0.0, 1.25, np.nan], dtype=np.float64)
    owners = np.array([0, 1, 2], dtype=np.int64)
    table = pa.table({"coordinate": coordinates, "owner": owners})
    path = tmp_path / "partition.parquet"

    pq.write_table(table, path)
    restored = pq.read_table(path)

    np.testing.assert_allclose(restored["coordinate"].to_numpy(), coordinates, equal_nan=True)
    np.testing.assert_array_equal(restored["owner"].to_numpy(), owners)
