import numpy as np

class DBSCAN:
    def __init__(self, eps: float = 0.5, min_pts: int = 3, period: float = 2*np.pi):
        self.eps = eps
        self.min_pts = min_pts
        self.num_of_clusters = 0
        self.period = period
        
    def euclidean_dist(self, a : np.ndarray, b : np.ndarray) -> float:
        return np.sqrt((a - b).dot(a - b))

    def max_angle_diff_dist(self, a : np.ndarray, b : np.ndarray) -> float:
        dists = np.abs(a - b) % self.period
        return np.max(np.minimum(dists, self.period - dists))

    def _pairwise_max_angle_dist_matrix(self, X: np.ndarray) -> np.ndarray:
        """Vectorized version of max_angle_diff_dist for all pairs at once."""
        diff = np.abs(X[:, None, :] - X[None, :, :]) % self.period
        diff = np.minimum(diff, self.period - diff)
        return diff.max(axis=2)  # [N, N]
        
    def fit_predict(self, X : np.ndarray) -> np.ndarray:
        n = X.shape[0]
        self.labels_ = -np.ones(n, int)

        dist_matrix = self._pairwise_max_angle_dist_matrix(X)
        neighbor_mask = dist_matrix <= self.eps
        np.fill_diagonal(neighbor_mask, False)

        for i in range(n):
            if self.labels_[i] != -1:
                continue
            neighbors = list(np.nonzero(neighbor_mask[i])[0])

            if (len(neighbors) + 1) < self.min_pts:
                continue

            self.labels_[i] = self.num_of_clusters
            queue = list(neighbors)
            visited_for_expansion = set()
            while queue:
                j = queue.pop()
                if self.labels_[j] != -1:
                    continue
                self.labels_[j] = self.num_of_clusters
                if j in visited_for_expansion:
                    continue
                visited_for_expansion.add(j)
                new_neighbors = np.nonzero(neighbor_mask[j])[0]
                if (len(new_neighbors) + 1) >= self.min_pts:
                    queue.extend(int(k) for k in new_neighbors if self.labels_[k] == -1)

            self.num_of_clusters += 1

        return self.labels_