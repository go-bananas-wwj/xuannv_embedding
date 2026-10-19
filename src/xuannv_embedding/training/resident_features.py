"""Keep verified frozen batches on their training device without repeated host conversion."""

from __future__ import annotations

import torch


class ResidentFeatures:
    """Materialize canonical streamed values once; preserve their dtype and exact bits."""

    def __init__(self, data, device: torch.device, *, chunk_size: int = 16) -> None:
        if type(chunk_size) is not int or chunk_size < 1 or data.n < 1:
            raise ValueError("Resident feature count and chunk size must be positive")
        self.n = data.n
        self.device = device
        self.resident_bytes = 0

        def allocate(value, count):
            if value is None:
                return None
            if isinstance(value, dict):
                return {k: allocate(v, count) for k, v in value.items()}
            if isinstance(value, tuple):
                return tuple(allocate(v, count) for v in value)
            if not isinstance(value, torch.Tensor) or value.ndim < 1 or len(value) != count:
                raise ValueError("Resident input must have an explicit batch dimension")
            self.resident_bytes += self.n * value[0].numel() * value.element_size()
            return torch.empty((self.n, *value.shape[1:]), dtype=value.dtype, device=device)

        def copy(dest, value, start, stop):
            if dest is None:
                if value is not None:
                    raise ValueError("Optional resident source changed between chunks")
                return
            if isinstance(dest, dict):
                if not isinstance(value, dict) or value.keys() != dest.keys():
                    raise ValueError("Resident input structure changed between chunks")
                for k in dest:
                    copy(dest[k], value[k], start, stop)
            elif isinstance(dest, tuple):
                if not isinstance(value, tuple) or len(value) != len(dest):
                    raise ValueError("Resident input structure changed between chunks")
                for d, v in zip(dest, value):
                    copy(d, v, start, stop)
            else:
                if value.shape != dest[start:stop].shape or value.dtype != dest.dtype:
                    raise ValueError("Resident input shape or dtype changed between chunks")
                dest[start:stop].copy_(value.detach().to(device))

        with torch.no_grad():
            for start in range(0, self.n, chunk_size):
                stop = min(self.n, start + chunk_size)
                batch = data.batch(list(range(start, stop)))
                if start == 0:
                    self.values = allocate(batch, stop - start)
                copy(self.values, batch, start, stop)

    def batch(self, positions):
        positions = list(positions)
        if not positions or any(type(v) is not int or v < 0 or v >= self.n for v in positions):
            raise ValueError("Resident batch positions must be nonempty valid tile indices")
        ids = torch.tensor(positions, dtype=torch.long, device=self.device)

        def gather(value):
            if isinstance(value, dict):
                return {k: gather(v) for k, v in value.items()}
            if isinstance(value, tuple):
                return tuple(gather(v) for v in value)
            return value.index_select(0, ids) if value is not None else None

        return gather(self.values)
