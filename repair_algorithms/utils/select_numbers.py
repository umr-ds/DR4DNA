"""Utility module for selecting unique numbers from sets.

This module provides efficient algorithms for selecting unique numbers from
multiple input sets. It uses a combination of greedy assignment and max-flow
(Dinic's algorithm) to find optimal assignments.

Main functions:
    select_numbers: Select up to n unique numbers per input set
"""

from collections import Counter, deque
from typing import Dict, Iterable, List, Set, Tuple


class _DinicFast:
    """A compact Dinic implementation optimized for Python: flattened edge arrays and local-variable access.
    Nodes are 0..n-1.
    """

    def __init__(self, n: int):
        """
        Initialize the Dinic max-flow solver.

        Args:
            n: Number of nodes in the graph
        """
        self.n = n
        self.adj: List[List[int]] = [[] for _ in range(n)]
        self.to: List[int] = []
        self.cap: List[int] = []
        self.rev: List[int] = []

    def add_edge(self, u: int, v: int, c: int) -> None:
        """
        Add a directed edge with capacity to the graph.

        Args:
            u: Source node index
            v: Target node index
            c: Edge capacity
        """
        # forward edge index
        fi = len(self.to)
        self.to.append(v)
        self.cap.append(c)
        self.rev.append(0)  # placeholder, will be set to index in v.adj
        self.adj[u].append(fi)
        # reverse edge index
        ri = len(self.to)
        self.to.append(u)
        self.cap.append(0)
        self.rev.append(0)
        self.adj[v].append(ri)
        # set reverse indices
        self.rev[fi] = ri
        self.rev[ri] = fi

    def _build_level_graph(self, s: int, t: int) -> List[int]:
        """Build level graph using BFS."""
        n = self.n
        level = [-1] * n
        q = deque([s])
        level[s] = 0
        to = self.to
        cap = self.cap
        adj = self.adj

        while q:
            u = q.popleft()
            for ei in adj[u]:
                v = to[ei]
                if cap[ei] > 0 and level[v] < 0:
                    level[v] = level[u] + 1
                    q.append(v)
        return level

    def _dfs_push(
        self,
        u: int,
        t: int,
        pushed: int,
        level: List[int],
        it: List[int],
    ) -> int:
        """DFS to push flow along augmenting paths."""
        if u == t or pushed == 0:
            return pushed

        ai = it[u]
        while ai < len(self.adj[u]):
            ei = self.adj[u][ai]
            v = self.to[ei]
            if self.cap[ei] > 0 and level[v] == level[u] + 1:
                tr = self._dfs_push(v, t, min(pushed, self.cap[ei]), level, it)
                if tr:
                    self.cap[ei] -= tr
                    self.cap[self.rev[ei]] += tr
                    it[u] = ai
                    return tr
            ai += 1
        it[u] = ai
        return 0

    def max_flow(self, s: int, t: int) -> int:
        """Calculate maximum flow from s to t using Dinic's algorithm."""
        flow = 0
        while True:
            level = self._build_level_graph(s, t)
            if level[t] < 0:
                break

            it = [0] * self.n
            pushed = self._dfs_push(s, t, 1 << 60, level, it)
            while pushed:
                flow += pushed
                pushed = self._dfs_push(s, t, 1 << 60, level, it)
        return flow


def _select_unique_numbers(
    sets: List[Set[int]], keys: List[int], cnt: Counter, n: int
) -> List[Tuple[int, Set[int]]]:
    """Select numbers that appear in exactly one set."""
    results: List[Set[int]] = []
    for i, s in enumerate(sets):
        uniques = [x for x in s if cnt[x] == 1]
        if len(uniques) >= n:
            results.append(set(sorted(uniques)[:n]))
        else:
            results.append(set())

    return [(keys[i], results[i]) for i in range(len(keys)) if len(results[i]) == n]


def _greedy_assign_unique_numbers(
    sets: List[Set[int]],
    cnt: Counter,
    needed: List[int],
    results: List[Set[int]],
) -> Dict[int, int]:
    """Greedy assignment of numbers that appear only once."""
    assigned: Dict[int, int] = {}

    for num in sorted([num for num, c in cnt.items() if c == 1]):
        for i, s in enumerate(sets):
            if num in s and needed[i] > 0:
                assigned[num] = i
                results[i].add(num)
                needed[i] -= 1
                break

    return assigned


def _greedy_assign_remaining_numbers(
    sets: List[Set[int]],
    rem_nums: List[int],
    needed: List[int],
    results: List[Set[int]],
    assigned: Dict[int, int],
) -> Dict[int, int]:
    """Greedy assignment of remaining numbers by frequency."""
    for num in rem_nums:
        if num in assigned:
            continue
        chosen = -1
        for i, s in enumerate(sets):
            if needed[i] > 0 and num in s:
                chosen = i
                break
        if chosen == -1:
            continue
        assigned[num] = chosen
        results[chosen].add(num)
        needed[chosen] -= 1

    return assigned


def _build_flow_network(
    sets: List[Set[int]],
    rem_nums: List[int],
    needed: List[int],
    assigned: Dict[int, int],
) -> Tuple[_DinicFast, int, int, int, int, List[int]]:
    """Build Dinic flow network for remaining assignments."""
    num_to_id: Dict[int, int] = {num: idx for idx, num in enumerate(rem_nums)}
    R = len(rem_nums)

    S = 0
    set_base = 1
    num_base = set_base + len(sets)
    T = num_base + R

    dinic = _DinicFast(T + 1)

    # Add edges from source to sets
    for i, need in enumerate(needed):
        if need > 0:
            dinic.add_edge(S, set_base + i, need)

    # Add edges from numbers to sink
    for j in range(R):
        dinic.add_edge(num_base + j, T, 1)

    # Add edges from sets to numbers
    for i, s in enumerate(sets):
        if needed[i] <= 0:
            continue
        for num in s:
            if num in num_to_id:
                dinic.add_edge(set_base + i, num_base + num_to_id[num], 1)

    return dinic, S, T, set_base, num_base, rem_nums


def _extract_flow_results(
    dinic: _DinicFast,
    results: List[Set[int]],
    set_base: int,
    num_base: int,
    rem_nums: List[int],
) -> None:
    """Extract assignment results from flow network."""
    to = dinic.to
    cap = dinic.cap
    adj = dinic.adj

    for j in range(len(rem_nums)):
        v = num_base + j
        for ei in adj[v]:
            u = to[ei]
            if set_base <= u < set_base + len(results) and cap[ei] > 0:
                set_idx = u - set_base
                results[set_idx].add(rem_nums[j])


def select_numbers(
    input_map: Dict[int, Iterable[int]],
    n: int,
    unique_only: bool = True,
    use_flow_fallback: bool = True,
    flow_threshold_slots: int = 10000,
) -> List[Tuple[int, Set[int]]]:
    """
    Fast greedy selection of up to `n` unique numbers per input set.

    Args:
        input_map: mapping from input id (int) to iterable of ints (converted to sets)
        n: required numbers per set (n >= 1)
        unique_only: if True, only use elements that appear in exactly one input set.
        use_flow_fallback: if True, use Dinic flow fallback to try to assign remaining slots.
        flow_threshold_slots: skip flow fallback if total remaining slots exceed this to avoid heavy computations.

    Returns:
        List of (input_id, set_of_n_numbers) for inputs that could be fulfilled.
    """
    if n <= 0:
        return []

    keys = list(input_map.keys())
    m = len(keys)
    if m == 0:
        return []

    sets = [set(input_map[k]) for k in keys]
    cnt: Counter = Counter()
    for s in sets:
        cnt.update(s)

    # Fast path: only unique numbers needed
    if unique_only:
        return _select_unique_numbers(sets, keys, cnt, n)

    results: List[Set[int]] = [set() for _ in range(m)]
    needed = [n] * m

    # Phase 1: Assign unique numbers
    assigned = _greedy_assign_unique_numbers(sets, cnt, needed, results)

    if all(k <= 0 for k in needed):
        return [(keys[i], results[i]) for i in range(m) if len(results[i]) == n]

    # Phase 2: Greedy assignment of remaining numbers
    rem_nums = [num for num in cnt if num not in assigned]
    rem_nums.sort(key=lambda x: (cnt[x], x))
    assigned = _greedy_assign_remaining_numbers(sets, rem_nums, needed, results, assigned)

    if all(k <= 0 for k in needed) or not use_flow_fallback:
        return [(keys[i], results[i]) for i in range(m) if len(results[i]) == n]

    # Phase 3: Flow-based assignment (if needed slots are manageable)
    total_slots = sum(max(0, k) for k in needed)
    if total_slots > flow_threshold_slots:
        return [(keys[i], results[i]) for i in range(m) if len(results[i]) == n]

    rem_nums = [num for num in rem_nums if num not in assigned]
    if len(rem_nums) == 0:
        return [(keys[i], results[i]) for i in range(m) if len(results[i]) == n]

    dinic, S, T, set_base, num_base, rem_nums = _build_flow_network(
        sets, rem_nums, needed, assigned
    )
    _ = dinic.max_flow(S, T)
    _extract_flow_results(dinic, results, set_base, num_base, rem_nums)

    return [(keys[i], results[i]) for i in range(m) if len(results[i]) == n]
