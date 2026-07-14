import numpy as np
import pytest

from olmo.data.dict_memmap_dataset import DictMemmapWriter


def test_writer_uses_bounded_index_file(tmp_path):
    writer = DictMemmapWriter(tmp_path, seq_len=2, file_seqs=4, max_entries=3)
    writer.write(
        np.asarray([4, 2, 7], dtype=np.int64),
        np.asarray([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=np.float32),
    )
    writer.close()

    index = np.memmap(tmp_path / "mmap_index.npy", dtype=np.int64, mode="r")
    assert len(index) == 3
    assert index.tolist() == [4, 2, 7]


def test_writer_rejects_rows_beyond_bounded_capacity(tmp_path):
    writer = DictMemmapWriter(tmp_path, seq_len=1, file_seqs=4, max_entries=1)
    with pytest.raises(RuntimeError, match="capacity exceeded"):
        writer.write(
            np.asarray([0, 1], dtype=np.int64),
            np.asarray([[1.0], [2.0]], dtype=np.float32),
        )


def test_writer_rejects_nonpositive_capacity(tmp_path):
    with pytest.raises(ValueError, match="must be positive"):
        DictMemmapWriter(tmp_path, max_entries=0)
