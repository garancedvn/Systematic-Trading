import numpy as np
from itertools import combinations

class CombinatorialPurgedCV:
    """
    CPCV de Prado (Advances in Financial ML, ch. 12).
    Découpe la série en N groupes contigus, et pour chaque combinaison de k
    groupes mis en test, le reste forme le train. Applique un embargo
    (purge) autour des frontières pour éviter le leakage label↔feature.

    Paramètres
    ----------
    n_splits : int — nombre de groupes (N)
    n_test_groups : int — k groupes en test par combinaison
    embargo_pct : float — fraction des données embargo de chaque côté du test
    """
    def __init__(self, n_splits=6, n_test_groups=2, embargo_pct=0.01):
        self.n_splits = n_splits
        self.n_test_groups = n_test_groups
        self.embargo_pct = embargo_pct

    def split(self, X, y=None, groups=None):
        n = len(X)
        embargo = int(n * self.embargo_pct)
        indices = np.arange(n)
        # Découpage en N groupes contigus
        group_bounds = np.array_split(indices, self.n_splits)

        for test_group_ids in combinations(range(self.n_splits), self.n_test_groups):
            test_idx = np.concatenate([group_bounds[g] for g in test_group_ids])
            test_set = set(test_idx)

            # Embargo : retirer les voisins immédiats des blocs test
            embargoed = set()
            for g in test_group_ids:
                lo = max(group_bounds[g][0] - embargo, 0)
                hi = min(group_bounds[g][-1] + embargo + 1, n)
                embargoed.update(range(lo, hi))

            train_idx = np.array(sorted(set(indices) - embargoed - test_set))
            yield train_idx, test_idx

    def get_n_splits(self, X=None, y=None, groups=None):
        from math import comb
        return comb(self.n_splits, self.n_test_groups)
