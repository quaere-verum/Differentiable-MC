from dataclasses import dataclass
import torch

@dataclass
class SimulationGrid:
    time_grid: torch.Tensor

    def merge(self, other: "SimulationGrid") -> "SimulationGrid":
        combined_grids = torch.cat((self.time_grid, other.time_grid))
        unique_times = torch.unique(combined_grids)
        unique_times_sorted = torch.sort(unique_times).values
        return SimulationGrid(unique_times_sorted)

    def find_times_in_grid(
        self,
        query_times: torch.Tensor,
        atol: float = 1e-8,
        rtol: float = 1e-6,
    ) -> torch.Tensor:

        if query_times.ndim != 1:
            raise ValueError("query_times must be 1-dimensional.")

        time_grid = self.time_grid

        if time_grid.ndim != 1:
            raise ValueError("time_grid must be 1-dimensional.")

        idx = torch.searchsorted(time_grid, query_times)

        if torch.any(idx >= time_grid.numel()):
            bad = query_times[idx >= time_grid.numel()]
            raise ValueError(
                f"Some query_times lie beyond the simulation grid: {bad.tolist()}"
            )

        grid_vals = time_grid[idx]

        matches = torch.isclose(grid_vals, query_times, atol=atol, rtol=rtol)

        if not torch.all(matches):
            bad = query_times[~matches]
            raise ValueError(
                f"Some query_times are not present in simulation grid: {bad.tolist()}"
            )

        return idx