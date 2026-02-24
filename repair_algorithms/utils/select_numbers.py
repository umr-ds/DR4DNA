from collections import Counter, deque
from typing import Iterable, List, Set, Tuple, Dict


class _DinicFast:
    """A compact Dinic implementation optimized for Python: flattened edge arrays and local-variable access.
    Nodes are 0..n-1.
    """

    def __init__(self, n: int):
        self.n = n
        self.adj: List[List[int]] = [[] for _ in range(n)]
        self.to: List[int] = []
        self.cap: List[int] = []
        self.rev: List[int] = []

    def add_edge(self, u: int, v: int, c: int) -> None:
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

    def max_flow(self, s: int, t: int) -> int:
        n = self.n
        flow = 0
        to = self.to
        cap = self.cap
        adj = self.adj
        rev = self.rev
        while True:
            level = [-1] * n
            q = deque([s])
            level[s] = 0
            while q:
                u = q.popleft()
                for ei in adj[u]:
                    v = to[ei]
                    if cap[ei] > 0 and level[v] < 0:
                        level[v] = level[u] + 1
                        q.append(v)
            if level[t] < 0:
                break
            it = [0] * n

            def dfs(u: int, pushed: int) -> int:
                if u == t or pushed == 0:
                    return pushed
                ai = it[u]
                while ai < len(adj[u]):
                    ei = adj[u][ai]
                    v = to[ei]
                    if cap[ei] > 0 and level[v] == level[u] + 1:
                        tr = dfs(v, pushed if pushed < cap[ei] else cap[ei])
                        if tr:
                            cap[ei] -= tr
                            cap[rev[ei]] += tr
                            it[u] = ai
                            return tr
                    ai += 1
                it[u] = ai
                return 0

            pushed = dfs(s, 1 << 60)
            while pushed:
                flow += pushed
                pushed = dfs(s, 1 << 60)
        return flow


def select_numbers(input_map: Dict[int, Iterable[int]], n: int, unique_only: bool = True,
                   use_flow_fallback: bool = True, flow_threshold_slots: int = 10000) -> List[Tuple[int, Set[int]]]:
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

    keys = list(input_map.keys())  # preserve dict insertion order
    m = len(keys)
    if m == 0:
        return []

    sets = [set(input_map[k]) for k in keys]

    cnt = Counter()
    for s in sets:
        cnt.update(s)

    results: List[Set[int]] = [set() for _ in range(m)]

    if unique_only:
        for i, s in enumerate(sets):
            uniques = [x for x in s if cnt[x] == 1]
            if len(uniques) >= n:
                results[i] = set(sorted(uniques)[:n])
        return [(keys[i], results[i]) for i in range(m) if len(results[i]) == n]

    needed = [n] * m
    assigned: Dict[int, int] = {}

    for num in sorted([num for num, c in cnt.items() if c == 1]):
        for i, s in enumerate(sets):
            if num in s and needed[i] > 0:
                assigned[num] = i
                results[i].add(num)
                needed[i] -= 1
                break

    if all(k <= 0 for k in needed):
        return [(keys[i], results[i]) for i in range(m) if len(results[i]) == n]

    rem_nums = [num for num in cnt if num not in assigned]
    rem_nums.sort(key=lambda x: (cnt[x], x))
    for num in rem_nums:
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

    if all(k <= 0 for k in needed) or not use_flow_fallback:
        return [(keys[i], results[i]) for i in range(m) if len(results[i]) == n]

    total_slots = sum(max(0, k) for k in needed)
    if total_slots > flow_threshold_slots:
        return [(keys[i], results[i]) for i in range(m) if len(results[i]) == n]

    rem_nums = [num for num in rem_nums if num not in assigned]
    num_to_id: Dict[int, int] = {num: idx for idx, num in enumerate(rem_nums)}
    R = len(rem_nums)
    if R == 0:
        return [(keys[i], results[i]) for i in range(m) if len(results[i]) == n]

    S = 0
    set_base = 1
    num_base = set_base + m
    T = num_base + R
    dinic = _DinicFast(T + 1)

    for i in range(m):
        need = needed[i]
        if need > 0:
            dinic.add_edge(S, set_base + i, need)

    for j in range(R):
        dinic.add_edge(num_base + j, T, 1)

    for i, s in enumerate(sets):
        if needed[i] <= 0:
            continue
        for num in s:
            if num in num_to_id:
                dinic.add_edge(set_base + i, num_base + num_to_id[num], 1)

    _ = dinic.max_flow(S, T)

    to = dinic.to
    cap = dinic.cap
    adj = dinic.adj
    for j in range(R):
        v = num_base + j
        for ei in adj[v]:
            u = to[ei]
            if set_base <= u < set_base + m and cap[ei] > 0:
                set_idx = u - set_base
                results[set_idx].add(rem_nums[j])

    return [(keys[i], results[i]) for i in range(m) if len(results[i]) == n]
