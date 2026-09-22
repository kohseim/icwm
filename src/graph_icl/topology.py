from __future__ import annotations
from dataclasses import dataclass
from functools import cached_property
from typing import Mapping
State = tuple[int, ...]

@dataclass(frozen=True)
class Topology:
    name: str
    dimension: int
    states: tuple[State, ...]
    neighbor_map: Mapping[State, tuple[State, ...]]

    def __post_init__(self) -> None:
        if self.dimension < 1:
            raise ValueError(f"unsupported dimension: {self.dimension}")
        if not self.states:
            raise ValueError("topology must contain at least one state")
        state_set = set(self.states)
        if len(state_set) != len(self.states):
            raise ValueError("states must be unique")
        for state in self.states:
            if len(state) != self.dimension:
                raise ValueError(f"state {state!r} has wrong dimension")
            if state not in self.neighbor_map:
                raise ValueError(f"missing neighbors for state {state!r}")
        for state, neighbors in self.neighbor_map.items():
            if state not in state_set:
                raise ValueError(f"neighbor map contains unknown state {state!r}")
            for neighbor in neighbors:
                if neighbor not in state_set:
                    raise ValueError(f"unknown neighbor {neighbor!r}")
                if state not in self.neighbor_map[neighbor]:
                    raise ValueError(f"edge {state!r}->{neighbor!r} is not symmetric")

    @cached_property
    def state_set(self) -> frozenset[State]:
        return frozenset(self.states)

    @cached_property
    def num_states(self) -> int:
        return len(self.states)

    @cached_property
    def right_offset(self) -> State:
        if self.name.startswith(("rectangular-grid-", "triangular-grid-")):
            return (0, 2)
        return (2, 0)


    @cached_property
    def edges(self) -> tuple[tuple[State, State], ...]:
        edges: set[tuple[State, State]] = set()
        for state, neighbors in self.neighbor_map.items():
            for neighbor in neighbors:
                edges.add(tuple(sorted((state, neighbor))))
        return tuple(sorted(edges))


    def neighbors(self, state: State) -> tuple[State, ...]:
        return self.neighbor_map[state]

    def are_adjacent(self, left: State, right: State) -> bool:
        return right in self.neighbor_map[left]


def grid(side_length: int) -> Topology:
    if side_length < 2:
        raise ValueError("grid topology requires side length at least two")

    states = tuple((i, j) for i in range(side_length) for j in range(side_length))
    deltas = ((-1, 0), (1, 0), (0, -1), (0, 1))
    neighbor_map: dict[State, tuple[State, ...]] = {}
    for i, j in states:
        neighbors: list[State] = []
        for d_i, d_j in deltas:
            candidate = (i + d_i, j + d_j)
            if 0 <= candidate[0] < side_length and 0 <= candidate[1] < side_length:
                neighbors.append(candidate)
        neighbor_map[(i, j)] = tuple(neighbors)
    return Topology(
        name=f"{side_length}x{side_length}-grid",
        dimension=2,
        states=states,
        neighbor_map=neighbor_map,
    )

def rectangular_grid(rows: int, cols: int) -> Topology:

    if rows < 2 or cols < 2:
        raise ValueError("rectangular grid dimensions must both be at least two")
    states = tuple((i, j) for i in range(rows) for j in range(cols))
    state_set = set(states)
    deltas = ((1, 0), (-1, 0), (0, 1), (0, -1))
    neighbor_map = {
        state: tuple(
            sorted(
                candidate
                for d_i, d_j in deltas
                if (candidate := (state[0] + d_i, state[1] + d_j)) in state_set
            )
        )
        for state in states
    }
    return Topology(
        name=f"rectangular-grid-{cols}x{rows}",
        dimension=2,
        states=states,
        neighbor_map=neighbor_map,
    )

def triangular_grid(rows: int, cols: int) -> Topology:

    if rows < 2 or cols < 2:
        raise ValueError("triangular grid dimensions must both be at least two")
    states = tuple((i, j) for i in range(rows) for j in range(cols))
    state_set = set(states)
    deltas = ((1, 0), (-1, 0), (0, 1), (0, -1), (1, -1), (-1, 1))
    neighbor_map = {
        state: tuple(
            sorted(
                candidate
                for d_i, d_j in deltas
                if (candidate := (state[0] + d_i, state[1] + d_j)) in state_set
            )
        )
        for state in states
    }
    return Topology(
        name=f"triangular-grid-{cols}x{rows}",
        dimension=2,
        states=states,
        neighbor_map=neighbor_map,
    )

def torus_grid(side_length: int) -> Topology:
    if side_length < 3:
        raise ValueError("torus grid topology requires side length at least three")
    states = tuple((i, j) for i in range(side_length) for j in range(side_length))
    deltas = ((-1, 0), (1, 0), (0, -1), (0, 1))
    neighbor_map: dict[State, tuple[State, ...]] = {}
    for i, j in states:
        neighbors = tuple(
            sorted(
                {
                    ((i + d_i) % side_length, (j + d_j) % side_length)
                    for d_i, d_j in deltas
                }
            )
        )
        neighbor_map[(i, j)] = neighbors
    return Topology(
        name=f"torus-{side_length}x{side_length}",
        dimension=2,
        states=states,
        neighbor_map=neighbor_map,
    )

TOPOLOGIES = {"lattice_4x4": grid(4), "lattice_5x5": grid(5), "lattice_3x5": rectangular_grid(3, 5), "triangle_4x4": triangular_grid(4, 4), "torus_4x4": torus_grid(4)}

def get_topology(name):
    if name in TOPOLOGIES:
        return TOPOLOGIES[name]
    for topology in TOPOLOGIES.values():
        if name == topology.name:
            return topology
    raise ValueError(f"Unsupported topology: {name}")

def adjacency_matrix(topology):
    return [[int(a != b and topology.are_adjacent(a, b)) for b in topology.states] for a in topology.states]
