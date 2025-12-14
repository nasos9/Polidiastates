import ast
import math
from datetime import datetime
from typing import List, Tuple, Dict, Any, Optional

import pandas as pd
import time

import lsh


# Assuming lsh_query is a function defined in a separate module 'lsh'
# from lsh import lsh_query

# Dummy lsh_query for code completeness if 'lsh' is not available
def lsh_query(keywords: List[str], num_neighbors: int, data: List[List[Any]], genre_index: int) -> List[Tuple[List[Any], float]]:
    """Placeholder for LSH query."""
    print(f"[LSH] Performing approximate nearest neighbor search on {len(data)} results with keywords: {keywords}")
    return [(row, 1.0) for row in data[:num_neighbors]]


def convert_release_date_to_numeric(date_val) -> int:
    """
    Convert release_date (string / datetime / NaN) to integer YYYYMMDD.
    If parsing fails, return 0.
    """
    if pd.isna(date_val):
        return 0
    try:
        dt = pd.to_datetime(str(date_val), errors="coerce")
        if pd.isna(dt):
            return 0
        return int(dt.year * 10000 + dt.month * 100 + dt.day)
    except Exception:
        return 0


class KDTreeNode:
    def __init__(self, point: Tuple[float, float, float], full_data: List[Any],
                 left: Optional["KDTreeNode"] = None, right: Optional["KDTreeNode"] = None):
        # point: 3D point [popularity, release_date_numeric, budget]
        self.point = point
        self.full_data = full_data
        self.left = left
        self.right = right


def build_kd_tree(points: List[Tuple[float, float, float]],
                  full_data: List[List[Any]],
                  depth: int = 0) -> Optional[KDTreeNode]:
    """
    Build a 3D k-d tree using:
        0: popularity
        1: release_date_numeric
        2: budget
    """
    if not points:
        return None

    k = 3
    axis = depth % k

    sorted_indices = sorted(range(len(points)), key=lambda i: points[i][axis])
    median = len(points) // 2

    return KDTreeNode(
        point=points[sorted_indices[median]],
        full_data=full_data[sorted_indices[median]],
        left=build_kd_tree(
            [points[i] for i in sorted_indices[:median]],
            [full_data[i] for i in sorted_indices[:median]],
            depth + 1,
        ),
        right=build_kd_tree(
            [points[i] for i in sorted_indices[median + 1:]],
            [full_data[i] for i in sorted_indices[median + 1:]],
            depth + 1,
        ),
    )


def range_query(node: Optional[KDTreeNode],
                range_min: List[float],
                range_max: List[float],
                depth: int = 0,
                results: Optional[List[List[Any]]] = None) -> List[List[Any]]:
    """
    Return all rows whose 3D point lies in:
        [range_min[0], range_max[0]] x
        [range_min[1], range_max[1]] x
        [range_min[2], range_max[2]]
    """
    if results is None:
        results = []
    if node is None:
        return results

    k = 3
    axis = depth % k

    if all(range_min[d] <= node.point[d] <= range_max[d] for d in range(k)):
        results.append(node.full_data)

    if node.point[axis] >= range_min[axis]:
        range_query(node.left, range_min, range_max, depth + 1, results)
    if node.point[axis] <= range_max[axis]:
        range_query(node.right, range_min, range_max, depth + 1, results)

    return results


def filter_by_categorical_inputs(results: List[List[Any]],
                                 categorical_inputs: Dict[str, List[str]],
                                 attribute_indices: Dict[str, int]) -> List[List[Any]]:
    """
    Filter rows based on categorical attributes.
    """
    import ast

    filtered_results: List[List[Any]] = []

    for row in results:
        match = True

        for attr, values in categorical_inputs.items():
            idx = attribute_indices[attr]
            search_values = [v.strip().lower() for v in values]

            if attr == "origin_country":
                raw = row[idx]
                try:
                    countries = ast.literal_eval(str(raw))
                except Exception:
                    countries = (
                        str(raw)
                        .replace("[", "")
                        .replace("]", "")
                        .replace("'", "")
                        .replace('"', "")
                        .split(",")
                    )
                tokens = [str(c).strip().lower() for c in countries if str(c).strip()]
                if not any(t in search_values for t in tokens):
                    match = False
                    break
            else:
                value = str(row[idx]).strip().lower()
                if value not in search_values:
                    match = False
                    break

        if match:
            filtered_results.append(row)

    return filtered_results

def kdtree_main(
    conditions: Optional[Dict[str, Any]] = None,
    genre_keywords: Optional[str] = None,
    num_of_results: Optional[int] = None
):

    conditions["release_date"] = tuple(convert_release_date_to_numeric(date) for date in conditions["release_date"])

    if conditions is None:
        conditions = {}

    # Variable to store execution time
    time_results: Dict[str, float] = {}

    # --- Load data ---
    data = pd.read_csv("movies_testing.csv")

    # --- Preprocessing ---
    # Ensure release_date_numeric exists
    if "release_date" in data.columns:
        data["release_date_numeric"] = data["release_date"].apply(convert_release_date_to_numeric)
    else:
        data["release_date_numeric"] = 0

    numeric_attributes = [
        "popularity",
        "budget",
        "revenue",
        "runtime",
        "vote_average",
        "vote_count",
        "release_date_numeric",
    ]
    categorical_attributes = ["original_language", "origin_country", "adult"]

    # convert numeric to float
    for col in numeric_attributes:
        if col in data.columns:
            data[col] = pd.to_numeric(data[col], errors="coerce").fillna(0.0)

    # --- Split conditions into numeric / categorical ---
    numeric_ranges: Dict[str, Tuple[float, float]] = {}
    categorical_inputs: Dict[str, List[str]] = {}

    for attr, value in conditions.items():
        key = attr
        if attr == "release_date":
            key = "release_date_numeric"

        if key in numeric_attributes:
            numeric_ranges[key] = value
        elif key in categorical_attributes:
            if isinstance(value, str):
                categorical_inputs[key] = [v.strip().lower() for v in value.split(",")]
            elif isinstance(value, list):
                categorical_inputs[key] = [v.strip().lower() for v in value]

    # --- 3D kd-tree columns ---
    columns_for_splitting = ["popularity", "release_date_numeric", "budget"]
    for col in columns_for_splitting:
        if col not in data.columns:
            raise ValueError(f"Column '{col}' not found")

    points = list(data[columns_for_splitting].to_records(index=False))
    full_data = data.values.tolist()

    # --- Build KD-tree ---
    print(f"[KD-TREE] Starting to build 3D KD-tree on {len(points)} points...")
    t_build_start = time.time()
    kd_tree = build_kd_tree(points, full_data)
    t_build_end = time.time()
    time_results['kdtree_build_time'] = t_build_end - t_build_start
    print(f"[KD-TREE] KD-tree built in: {time_results['kdtree_build_time']:.6f} seconds")

    # --- Build range_min/max ---
    range_min: List[float] = []
    range_max: List[float] = []
    for col in columns_for_splitting:
        if col in numeric_ranges:
            min_val, max_val = numeric_ranges[col]
            range_min.append(min_val if min_val is not None else -math.inf)
            range_max.append(max_val if max_val is not None else math.inf)
        else:
            range_min.append(-math.inf)
            range_max.append(math.inf)

    # Ignore date filtering if all dates are missing/zero
    if "release_date_numeric" in data.columns:
        if data["release_date_numeric"].min() == 0 and data["release_date_numeric"].max() == 0:
            idx_date = columns_for_splitting.index("release_date_numeric")
            range_min[idx_date] = -math.inf
            range_max[idx_date] = math.inf

    print(f"[QUERY] Starting search process (KD-tree range query, categorical filters, optional LSH)...")
    t_start = time.time()

    # --- KD-tree range search ---
    print(f"[KD-TREE] Performing range query with ranges: {list(zip(columns_for_splitting, range_min, range_max))}")
    search_results = range_query(kd_tree, range_min, range_max)
    print(f"[KD-TREE] Found {len(search_results)} results in range query.")

    columns = ["id", "title", "adult", "original_language", "origin_country", "release_date", "genre_names", "production_company_names", "budget", "revenue", "runtime", "popularity", "vote_average", "vote_count"]
    query_results_dict = [dict(zip(columns, row)) for row in search_results]
    search_results_df = pd.DataFrame(query_results_dict)

    if genre_keywords:  # perform lsh only if there is a genre keyword
        search_results_df['genre_names'] = search_results_df['genre_names'].apply(ast.literal_eval)  # cast to a list
        search_results_df['genre_names_cleaned'] = search_results_df['genre_names'].apply(lambda x: " ".join(x).lower())

        buckets = lsh.lsh(search_results_df)
        candidates = lsh.lsh_query(buckets, genre_keywords)

        filtered_movies = search_results_df[search_results_df["id"].isin(candidates)].head(num_of_results)
        results = filtered_movies.values.tolist()

    else:
        results = search_results_df.values.tolist()

    t_end = time.time()
    time_results['total_query_time'] = t_end - t_start
    print(f"[QUERY] Total Query time (range query + filters): {time_results['total_query_time']:.6f} seconds")

    return results
