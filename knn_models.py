import heapq
from collections import Counter
import numpy as np


class BaseKNN:
    """Abstract base class for k-NN estimators implementing scikit-style API."""

    def __init__(self, k=5):
        self.k = k
        self.X = None
        self.y = None

    def fit(self, X, y=None):
        raise NotImplementedError

    def query(self, q, k=None):
        raise NotImplementedError

    def predict_one(self, q):
        _, idxs = self.query(q)
        votes = Counter(self.y[i] for i in idxs)
        top = max(votes.values())
        tied = {lbl for lbl, c in votes.items() if c == top}
        for i in idxs:
            if self.y[i] in tied:
                return self.y[i]

    def predict(self, Q):
        return np.array([self.predict_one(q) for q in Q])

    def score(self, Q, y_true):
        return float(np.mean(self.predict(Q) == np.asarray(y_true)))


class BruteForceKNN(BaseKNN):
    """Vectorized NumPy Brute-Force baseline."""

    def fit(self, X, y=None):
        self.X = np.asarray(X, dtype=float)
        self.y = None if y is None else np.asarray(y)
        return self

    def query(self, q, k=None):
        k = min(k or self.k, len(self.X))
        diff = self.X - q
        dists_sq = np.sum(diff * diff, axis=1)

        top_k_idx = np.argpartition(dists_sq, k - 1)[:k]
        top_k_idx = top_k_idx[np.argsort(dists_sq[top_k_idx])]

        return np.sqrt(dists_sq[top_k_idx]), top_k_idx


class KDTreeNode:
    """Spatial node storing split boundaries or leaf point arrays."""

    def __init__(
        self,
        axis=None,
        split_val=None,
        left=None,
        right=None,
        is_leaf=False,
        points=None,
        labels=None,
        indices=None,
    ):
        self.axis = axis
        self.split_val = split_val
        self.left = left
        self.right = right
        self.is_leaf = is_leaf
        self.points = points
        self.labels = labels
        self.indices = indices


class KDTree:
    """Spatial KD-Tree with vectorized leaf buckets and O(D log N) search pruning."""

    def __init__(self, X, y, leaf_size=32):
        self.X = np.asarray(X, dtype=float)
        self.y = np.asarray(y)
        self.leaf_size = leaf_size
        self.n_samples, self.n_features = self.X.shape
        self.root = self.build_tree()

    def build_tree(self):
        indices = np.arange(self.n_samples)
        return self._build(indices, depth=0)

    def _build(self, indices, depth):
        if len(indices) <= self.leaf_size:
            return KDTreeNode(
                is_leaf=True,
                points=self.X[indices],
                labels=self.y[indices],
                indices=indices,
            )

        axis = depth % self.n_features
        median_pos = len(indices) // 2

        order = np.argpartition(self.X[indices, axis], median_pos)
        sorted_indices = indices[order]
        split_val = self.X[sorted_indices[median_pos], axis]

        left = self._build(sorted_indices[:median_pos], depth + 1)
        right = self._build(sorted_indices[median_pos:], depth + 1)

        return KDTreeNode(
            axis=axis, split_val=split_val, left=left, right=right
        )

    def query(self, query_point, k=5):
        query_point = np.asarray(query_point, dtype=float)
        k = min(k, self.n_samples)
        nearest = []  # Max-heap: (-dist_sq, index, label)

        def search(node):
            if node is None:
                return

            if node.is_leaf:
                diff = node.points - query_point
                dists_sq = np.sum(diff * diff, axis=1)

                for i in range(len(dists_sq)):
                    d_sq = float(dists_sq[i])
                    idx = int(node.indices[i])
                    lbl = node.labels[i]

                    if len(nearest) < k:
                        heapq.heappush(nearest, (-d_sq, idx, lbl))
                    elif d_sq < -nearest[0][0]:
                        heapq.heapreplace(nearest, (-d_sq, idx, lbl))
                return

            axis = node.axis
            axis_diff = query_point[axis] - node.split_val

            near_branch, far_branch = (
                (node.left, node.right)
                if axis_diff <= 0
                else (node.right, node.left)
            )

            search(near_branch)

            if len(nearest) < k or (axis_diff * axis_diff) < -nearest[0][0]:
                search(far_branch)

        search(self.root)

        results = [(np.sqrt(-item[0]), item[2], item[1]) for item in nearest]
        results.sort(key=lambda x: (x[0], x[2]))
        return results

    def print_tree_structure(self, node=None, depth=0, max_depth=3):
        """Prints a textual representation showing spatial partitioning down to max_depth."""
        if node is None:
            node = self.root

        indent = "  " * depth
        if node.is_leaf:
            return f"{indent}└── [Leaf Bucket] Points: {len(node.indices)}\n"

        summary = f"{indent}├── [Depth {depth}] Axis {node.axis} Split Threshold: {node.split_val:.4f}\n"

        if depth < max_depth:
            if node.left:
                summary += self.print_tree_structure(node.left, depth + 1, max_depth)
            if node.right:
                summary += self.print_tree_structure(node.right, depth + 1, max_depth)
        else:
            summary += f"{indent}  ├── Left Subtree (...)\n{indent}  └── Right Subtree (...)\n"

        return summary


class KDTreeKNN(BaseKNN):
    """Standard Scikit-style estimator wrapping custom KD-Tree."""

    def __init__(self, k=5, leaf_size=32):
        super().__init__(k=k)
        self.leaf_size = leaf_size
        self.tree = None

    def fit(self, X, y):
        self.X = np.asarray(X, dtype=float)
        self.y = np.asarray(y)
        self.tree = KDTree(self.X, self.y, leaf_size=self.leaf_size)
        return self

    def query(self, query_point, k=None):
        k = k or self.k
        results = self.tree.query(query_point, k=k)
        dists = [item[0] for item in results]
        indices = [item[2] for item in results]
        return dists, indices


def validate_retrieval_precision(n_samples=5000, n_features=4, n_queries=100, decimal_precision=5):
    """Automated assertion suite matching KD-Tree retrieval against Brute-Force to 5 decimal places."""
    from sklearn.datasets import make_classification

    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=4,
        n_redundant=0,
        random_state=42,
    )

    kd_tree = KDTree(X, y, leaf_size=16)
    bf_model = BruteForceKNN(k=5).fit(X, y)

    for i in range(n_queries):
        query_vec = X[i]
        kd_res = kd_tree.query(query_vec, k=5)
        kd_dists = np.array([item[0] for item in kd_res])

        bf_dists, _ = bf_model.query(query_vec, k=5)

        np.testing.assert_array_almost_equal(kd_dists, bf_dists, decimal=decimal_precision)

    return True