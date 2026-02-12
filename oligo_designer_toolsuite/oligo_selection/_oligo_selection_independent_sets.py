############################################
# imports
############################################

import gc

import networkx as nx
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, lil_matrix

from oligo_designer_toolsuite.database import OligoDatabase
from oligo_designer_toolsuite.oligo_efficiency_filter import OligoScoring, SetScoringBase

from ._oligo_selection_base import BaseOligoSelection

############################################
# Oligoset Selection Classes
############################################


class IndependentSetsOligoSelection(BaseOligoSelection):
    """
    Generates multiple non-overlapping sets of oligos using a graph-based selection strategy.
    Optimizes sets according to per-oligo and set-level scoring, while respecting minimum/maximum
    set size and minimum distance between oligos. Uses a compatibility graph (non-overlap matrix)
    and clique-finding to select diverse, high-scoring sets.

    :param oligos_scoring: Scoring method for individual oligos.
    :type oligos_scoring: OligoScoring
    :param set_scoring: Scoring method for evaluating the quality of each oligo set.
    :type set_scoring: SetScoringBase
    :param set_size_opt: The optimal size of each oligo set.
    :type set_size_opt: int
    :param set_size_min: The minimum allowed size of each oligo set.
    :type set_size_min: int
    :param distance_between_oligos: Minimum allowed distance between oligos in the set.
    :type distance_between_oligos: int, optional
    :param n_attempts_graph: Number of randomized graph attempts used during oligo set generation.
        In each attempt, a fraction of nodes is randomly removed from the compatibility graph to
        create a perturbed graph, from which candidate cliques (oligo sets) are enumerated.
        Increasing this value increases diversity among candidate sets at the cost of runtime.
    :type n_attempts_graph: int
    :param n_attempts_clique_enum: Maximum number of cliques enumerated per graph attempt.
        Clique enumeration is performed using a deterministic algorithm, and this parameter
        limits how many cliques are explored before stopping enumeration for the current graph.
        Increasing this value allows exploration of more candidate oligo sets within each graph
        attempt but increases runtime.
    :type n_attempts_clique_enum: int
    :param diversification_fraction: Fraction of oligos to remove at random per attempt to create diversity. Default is 0.1.
    :type diversification_fraction: float, optional
    """

    def __init__(
        self,
        oligos_scoring: OligoScoring,
        set_scoring: SetScoringBase,
        set_size_opt: int,
        set_size_min: int,
        distance_between_oligos: int,
        n_attempts_graph: int,
        n_attempts_clique_enum: int,
        diversification_fraction: float,
        jaccard_opt: float,
        jaccard_step: float,
    ) -> None:
        """Constructor for the IndependentSetsOligoSelection class."""
        self.oligos_scoring = oligos_scoring
        self.set_scoring = set_scoring
        self.set_size_opt = set_size_opt
        self.set_size_min = set_size_min
        self.distance_between_oligos = distance_between_oligos
        self.n_attempts_graph = n_attempts_graph
        self.n_attempts_clique_enum = n_attempts_clique_enum
        self.diversification_fraction = diversification_fraction
        self.jaccard_opt = jaccard_opt
        self.jaccard_step = jaccard_step
        if jaccard_opt < 0 or jaccard_opt > 1:
            raise ValueError("jaccard_opt must be between 0 and 1")
        if jaccard_step < 0 or jaccard_step > 1:
            raise ValueError("jaccard_step must be between 0 and 1")

    def _get_oligo_sets_for_region(
        self,
        oligo_database: OligoDatabase,
        sequence_type: str,
        region_id: str,
        n_sets: int,
    ) -> None:
        """
        Computes the oligo set for a specific region by building a non-overlap matrix, then
        selecting oligo sets via the graph-based (clique) strategy. Optionally prunes the
        database to keep only oligos that appear in the selected sets.

        :param oligo_database: The OligoDatabase instance containing oligonucleotide sequences and
            their associated properties. This database stores oligo data organized by genomic regions.
        :type oligo_database: OligoDatabase
        :param sequence_type: Type of sequence being processed.
        :type sequence_type: str
        :param region_id: Region ID to process.
        :type region_id: str
        :param n_sets: The number of oligo sets to generate.
        :type n_sets: int
        """
        # create the overlapping matrix
        non_overlap_matrix, non_overlap_matrix_ids = self._get_non_overlap_matrix(
            oligo_database=oligo_database, region_id=region_id
        )

        # pre-filter oligos by degree
        non_overlap_matrix, non_overlap_matrix_ids = self._pre_filter_oligos_by_degree(
            non_overlap_matrix=non_overlap_matrix,
            non_overlap_matrix_ids=non_overlap_matrix_ids,
            min_oligoset_size=self.set_size_min,
        )

        # generate candidate sets
        if len(non_overlap_matrix_ids) >= self.set_size_min:
            oligosets = self._generate_candidate_sets(
                oligo_database=oligo_database,
                region_id=region_id,
                sequence_type=sequence_type,
                non_overlap_matrix=non_overlap_matrix,
                non_overlap_matrix_ids=non_overlap_matrix_ids,
            )
        else:
            oligosets = {}

        # select diverse oligosets
        oligosets = self._select_diverse_oligosets(
            oligosets=oligosets,
            n_sets=n_sets,
        )

        # remove oligos from database that are not part of oligosets
        oligos_keep = set().union(*oligosets.keys())
        oligo_ids = list(oligo_database.database[region_id].keys())
        for oligo_id in oligo_ids:
            if oligo_id not in oligos_keep:
                del oligo_database.database[region_id][oligo_id]

        # format oligosets to dataframe
        oligo_database.oligosets[region_id] = self._oligosets_to_dataframe(oligosets)

        # delete unused variables to free some memory
        del non_overlap_matrix
        gc.collect()

    def _get_non_overlap_matrix(
        self,
        oligo_database: OligoDatabase,
        region_id: str,
    ) -> tuple[csr_matrix, list[str]]:
        """
        Generates a sparse matrix that represents the overlap between oligos in the specified region of the oligo database.
        The matrix is computed based on the intervals (start, end) of each oligo, with a distance threshold to determine overlap.
        The matrix has dimensions n_oligos * n_oligos. Each entry contains 1 if the correspondent oligos don't overlap and 0 if they overlap.

        :param oligo_database: The OligoDatabase instance containing oligonucleotide sequences and their associated properties. This database stores oligo data organized by genomic regions and can be used for filtering, property calculations, set generation, and output operations.
        :type oligo_database: OligoDatabase
        :param region_id: Region ID to process.
        :type region_id: str
        :return: A sparse matrix representing the non-overlap between oligos, and a list of oligo IDs.
        :rtype: tuple(csr_matrix, list)
        """

        def _get_distance(seq1_intervals: list[list[int]], seq2_intervals: list[list[int]]) -> int:
            # Determine if two ligos do NOT overlap based on a distance value
            distances = min(
                [max(a[0], b[0]) - min(a[1], b[1]) for a in seq1_intervals for b in seq2_intervals]
            )
            if distances <= self.distance_between_oligos:
                return 0
            return int(distances)

        # Keep track of the indices
        non_overlap_matrix_ids = list(oligo_database.database[region_id].keys())

        # Get all intervals (start, end)
        intervals = []
        for oligo_id in non_overlap_matrix_ids:
            intervals.append(
                [
                    [s, e]
                    # loop through sequences from different genomic regions for the same oligo (can have the same coordinates if coming from shorter and longer exons)
                    for start, end in zip(
                        oligo_database.database[region_id][oligo_id]["start"],
                        oligo_database.database[region_id][oligo_id]["end"],
                    )
                    # loop through exon junction parts for the same oligo
                    for s, e in zip(start, end)
                ]
            )

        # Create a sparse non-overlap matrix
        n_oligos = len(non_overlap_matrix_ids)
        non_overlap_matrix = lil_matrix((n_oligos, n_oligos), dtype=int)

        # Calculate only upper triangle matrix since the matrix is symmetric
        for i in range(n_oligos):
            for j in range(i + 1, n_oligos):
                non_overlap_matrix[i, j] = _get_distance(intervals[i], intervals[j])

        # Fill values of lower triangle
        non_overlap_matrix += non_overlap_matrix.T
        non_overlap_matrix = non_overlap_matrix.tocsr()

        return non_overlap_matrix, non_overlap_matrix_ids

    def _pre_filter_oligos_by_degree(
        self,
        non_overlap_matrix: csr_matrix,
        non_overlap_matrix_ids: list[str],
        min_oligoset_size: int,
    ) -> tuple[csr_matrix, list[str]]:
        """
        Pre-filters oligos by removing nodes that cannot participate in any oligo set of size
        at least min_oligoset_size due to insufficient degree in the compatibility graph.

        A necessary condition for a node to be in a k-sized compatible set (clique) is
        degree(node) >= k - 1. Therefore only nodes with degree >= (min_oligoset_size - 1)
        are kept.

        :param non_overlap_matrix: Sparse matrix of non-overlap (compatibility) between oligos.
        :type non_overlap_matrix: csr_matrix
        :param non_overlap_matrix_ids: List of oligo IDs corresponding to the matrix rows/columns.
        :type non_overlap_matrix_ids: list[str]
        :param min_oligoset_size: Minimum required oligo set size; nodes with degree < min_oligoset_size - 1 are removed.
        :type min_oligoset_size: int
        :return: The filtered non-overlap matrix and the corresponding list of oligo IDs.
        :rtype: tuple[csr_matrix, list[str]]
        """
        # get degrees of each oligo
        degrees = non_overlap_matrix.getnnz(axis=1)
        min_degree = min_oligoset_size - 1

        # keep oligos with degree >= min_degree
        nodes_to_keep = np.where(degrees >= min_degree)[0].astype(int)

        # if no oligos with degree >= min_degree, return empty matrix and ids
        if nodes_to_keep.size == 0:
            return non_overlap_matrix[:0, :0].tocsr(), []

        # filter matrix and ids
        non_overlap_matrix_filt = non_overlap_matrix[nodes_to_keep, :][:, nodes_to_keep].tocsr()
        non_overlap_matrix_ids_filt = [non_overlap_matrix_ids[i] for i in nodes_to_keep.tolist()]

        return non_overlap_matrix_filt, non_overlap_matrix_ids_filt

    def _generate_candidate_sets(
        self,
        oligo_database: OligoDatabase,
        region_id: str,
        sequence_type: str,
        non_overlap_matrix: csr_matrix,
        non_overlap_matrix_ids: list[str],
        seed: int = 42,
    ) -> dict[tuple[str, ...], dict[str, float]]:
        """
        Generates candidate oligo sets by finding cliques in the compatibility graph and scoring them.

        Builds a graph from the non-overlap matrix, finds an initial maximal clique with a greedy
        heuristic, then repeatedly removes a random fraction of nodes and enumerates cliques to
        discover diverse candidate sets. Each set is scored using the oligo and set scoring
        methods. Returns a dict mapping oligo tuple (set) to its score dict.

        :param oligo_database: The OligoDatabase instance containing oligo sequences and properties.
        :type oligo_database: OligoDatabase
        :param region_id: Region ID to process.
        :type region_id: str
        :param sequence_type: Type of sequence being processed.
        :type sequence_type: str
        :param non_overlap_matrix: Sparse compatibility matrix (non-zero = oligos can be in the same set).
        :type non_overlap_matrix: csr_matrix
        :param non_overlap_matrix_ids: List of oligo IDs corresponding to the matrix indices.
        :type non_overlap_matrix_ids: list[str]
        :param seed: Random seed for the diversification step (random node removal). Default is 42.
        :type seed: int, optional
        :return: Dict mapping tuple of oligo IDs to their set score dict.
        :rtype: dict[tuple[str, ...], dict[str, float]]
        """

        def _add_clique_to_oligosets(clique: list[str], oligoset_size: int) -> None:
            oligos_scores = self.oligos_scoring.apply(
                oligo_database=oligo_database,
                region_id=region_id,
                oligo_ids=clique,
                sequence_type=sequence_type,
                non_overlap_matrix=non_overlap_matrix,
                non_overlap_matrix_ids=non_overlap_matrix_ids,
                set_oligo_ids=clique,
                oligoset_size=oligoset_size,
            )
            oligoset, oligoset_scores = self.set_scoring.apply(oligos_scores, oligoset_size)
            oligosets[tuple(sorted(oligoset))] = oligoset_scores

        oligosets: dict[tuple[str, ...], dict[str, float]] = {}
        oligoset_size = self.set_size_min

        rng = np.random.default_rng(seed)
        n_nodes_removed = int(self.diversification_fraction * len(non_overlap_matrix_ids))

        for attempt in range(self.n_attempts_graph):

            # --- Build full graph ---
            G = nx.from_scipy_sparse_array(non_overlap_matrix)
            G = nx.relabel_nodes(G, dict(enumerate(non_overlap_matrix_ids)))

            # --- Diversification: remove nodes (except first attempt) ---
            if attempt > 0 and n_nodes_removed > 0:
                # Weight node removal by oligo scores to bias diversification.
                # alpha controls the strength of this bias:
                #   alpha → 0 : nearly uniform random removal
                #   alpha → 1 : increasingly biased by oligo scores
                alpha = attempt / self.n_attempts_graph

                oligos_scores = self.oligos_scoring.apply(
                    oligo_database=oligo_database,
                    region_id=region_id,
                    oligo_ids=list(G.nodes),
                    sequence_type=sequence_type,
                    non_overlap_matrix=non_overlap_matrix,
                    non_overlap_matrix_ids=non_overlap_matrix_ids,
                    set_oligo_ids=None,
                    oligoset_size=None,
                )

                weights = oligos_scores**alpha
                total = weights.sum()
                weights = np.ones_like(weights) / len(weights) if total == 0 else weights / total

                nodes_to_remove = rng.choice(
                    list(G.nodes),
                    size=n_nodes_removed,
                    replace=False,
                    p=weights,
                )

                G.remove_nodes_from(list(nodes_to_remove))

            # --- Check feasibility via greedy clique ---
            greedy_max_clique = self._greedy_max_clique(G)

            if len(greedy_max_clique) < oligoset_size:
                if attempt == 0:
                    break  # even full graph cannot support min size
                continue

            oligoset_size = min(self.set_size_opt, len(greedy_max_clique))
            _add_clique_to_oligosets(greedy_max_clique, oligoset_size)

            # --- Enumerate cliques ---
            for i, clique in enumerate(nx.find_cliques(G)):
                if i >= self.n_attempts_clique_enum:
                    break
                if len(clique) < oligoset_size:
                    continue

                _add_clique_to_oligosets(clique, oligoset_size)

            del G
            gc.collect()

        return oligosets

    def _greedy_max_clique(self, G: nx.Graph) -> list[str]:
        """
        Finds a maximal clique in the graph using a greedy degree-based heuristic.

        Nodes are processed in descending order of degree. A node is added to the clique
        if it is adjacent to all nodes currently in the clique. The result is guaranteed
        to be a valid maximal clique but not necessarily a maximum-size clique.

        :param G: The compatibility graph (nodes = oligo IDs, edges = compatible pairs).
        :type G: nx.Graph
        :return: List of node IDs (oligo IDs) in the computed clique.
        :rtype: list[str]
        """
        nodes = sorted(G.nodes(), key=lambda n: G.degree(n), reverse=True)
        clique: list[str] = []
        for n in nodes:
            if all(G.has_edge(n, c) for c in clique):
                clique.append(n)
        return clique

    def _select_diverse_oligosets(
        self,
        oligosets: dict[tuple[str, ...], dict[str, float]],
        n_sets: int,
    ) -> dict[tuple[str, ...], dict[str, float]]:
        """
        Selects diverse oligo sets from a list of candidate sets.

        :param oligosets: Dictionary of candidate sets.
        :type oligosets: dict[tuple[str, ...], dict[str, float]]
        :param n_sets: Number of sets to select.
        :type n_sets: int
        :return: Dictionary of selected sets.
        :rtype: dict[tuple[str, ...], dict[str, float]]
        """
        if not oligosets:
            return {}
        score_names = self.set_scoring.score_names
        ascending = self.set_scoring.ascending

        # Sort oligosets by score
        items = list(oligosets.items())
        items.sort(
            key=lambda kv: tuple(kv[1][s] for s in score_names),
            reverse=not ascending,
        )

        # Greedy Jaccard-based selection with adaptive relaxation
        selected: dict[tuple[str, ...], dict[str, float]] = {}
        selected_sets: list[set[str]] = []

        current_jaccard = self.jaccard_opt

        while current_jaccard <= 1.0 and len(selected) < n_sets:
            for oligos, scores in items:
                if oligos in selected:
                    continue

                S = set(oligos)

                if all(len(S & T) / len(S) <= current_jaccard for T in selected_sets):
                    selected[oligos] = scores
                    selected_sets.append(S)

                if len(selected) >= n_sets:
                    break

            if self.jaccard_step == 0:
                break
            elif self.jaccard_step > 0:
                current_jaccard = round(current_jaccard + self.jaccard_step, 3)

        return selected

    def _oligosets_to_dataframe(
        self,
        oligosets: dict[tuple[str, ...], dict[str, float]],
    ) -> pd.DataFrame:
        """
        Formats the list of candidate oligo sets into a DataFrame, deduplicates by oligo composition,
        sorts by the set score columns, and greedily selects up to n_sets sets while enforcing a
        Jaccard diversity constraint so that selected sets are not too similar. If not enough sets
        are found, the Jaccard threshold is relaxed in steps.

        :param oligosets: List of candidate sets; each element is a list of oligo IDs followed by score values.
        :type oligosets: dict[tuple[str, ...], dict[str, float]]
        :return: DataFrame with columns oligoset_id, oligo_0, ..., oligo_n, and the set score columns.
        :rtype: pd.DataFrame
        """
        if not oligosets:
            return pd.DataFrame()
        score_names = self.set_scoring.score_names
        oligos = list(oligosets.keys())
        oligoset_size = len(oligos[0])

        rows = []
        for oligoset, scores in oligosets.items():
            rows.append(list(oligoset) + [scores[s] for s in score_names])

        columns = [f"oligo_{i}" for i in range(oligoset_size)] + score_names

        df = pd.DataFrame(rows, columns=columns)
        df.insert(0, "oligoset_id", range(len(df)))
        return df
