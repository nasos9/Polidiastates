import ast
from datetime import datetime

import pandas as pd
import time
import lsh


class RangeTreeNode:
    def __init__(self, record=None):
        self.record = record  # Movie record stored at this node (D1)
        self.left = None
        self.right = None
        self.subtree_struct = None  # (d-1)-dimensional range tree

def build_1d_tree(records, dim):
    if not records:
        return None

    records.sort(key=lambda p: p[dim])
    mid = len(records) // 2
    node = RangeTreeNode(records[mid])

    node.left = build_1d_tree(records[:mid], dim)
    node.right = build_1d_tree(records[mid + 1:], dim)

    return node

def build_range_tree(records, dims):
    if not records:
        return None

    if len(dims) == 1:
        return build_1d_tree(records, dims[0])

    dim = dims[0]
    records_sorted = sorted(records, key=lambda p: p[dim])
    mid = len(records_sorted) // 2
    node = RangeTreeNode(records_sorted[mid])

    left_records = records_sorted[:mid]
    right_records = records_sorted[mid + 1:]

    node.left = build_range_tree(left_records, dims)
    node.right = build_range_tree(right_records, dims)

    subtree_records = records_sorted
    node.subtree_struct = build_range_tree(subtree_records, dims[1:])

    return node

def in_range(record, dims, ranges):
    for dim, (low, high) in zip(dims, ranges):
        val = record[dim]
        if val < low or val > high:
            return False
    return True

def query_1d(node, dim, low, high, output):
    if not node:
        return

    value = node.record[dim]

    if value >= low:
        query_1d(node.left, dim, low, high, output)

    if low <= value <= high:
        output.append(node.record)

    if value <= high:
        query_1d(node.right, dim, low, high, output)

def range_query(node, dims, ranges, output):
    # 1D tree
    if len(dims) == 1:
        dim = dims[0]
        low, high = ranges[0]
        query_1d(node, dim, low, high, output)
        return

    primary_dim = dims[0]
    low, high = ranges[0]

    # find split node
    split = node
    while split and not (low <= split.record[primary_dim] <= high):
        if split.record[primary_dim] < low:
            split = split.right
        else:
            split = split.left
    if not split:
        return

    # left path of split node
    current_node = split.left
    while current_node:
        if current_node.record[primary_dim] >= low:
            if current_node.right:
                range_query(current_node.right.subtree_struct, dims[1:], ranges[1:], output)

            if in_range(current_node.record, dims, ranges):
                output.append(current_node.record)

            current_node = current_node.left
        else:
            current_node = current_node.right

    # right path of split node
    current_node = split.right
    while current_node:
        if current_node.record[primary_dim] <= high:
            if current_node.left:
                range_query(current_node.left.subtree_struct, dims[1:], ranges[1:], output)

            if in_range(current_node.record, dims, ranges):
                output.append(current_node.record)

            current_node = current_node.right
        else:
            current_node = current_node.left

    # check split node
    if in_range(split.record, dims, ranges):
        output.append(split.record)

def search_range_tree(root_node, query_dictionary):
    dims = list(query_dictionary.keys())
    ranges = list(query_dictionary.values())
    output = []

    range_query(root_node, dims, ranges, output)

    return output

def brute_force_range_query(records, query_dictionary):
    dims = list(query_dictionary.keys())
    ranges = list(query_dictionary.values())
    results = []

    for record in records:
        if in_range(record, dims, ranges):
            results.append(record)
    return results

def convert_date_to_numeric(date_str):
    return int(date_str.replace('-', ''))

def numeric_to_date(numeric_date):
    s = str(numeric_date)  # convert to string
    year = s[:4]
    month = s[4:6]
    day = s[6:8]
    return f"{year}-{month}-{day}"

def range_tree_main(conditions, genre_kw, num_of_results):

    conditions["release_date"] = tuple(convert_date_to_numeric(date) for date in conditions["release_date"])
    dimensions = list(conditions.keys())

    movies = pd.read_csv("movies_testing.csv")
    movies['release_date'] = movies['release_date'].apply(convert_date_to_numeric)

    movies = movies.to_dict(orient="records")
    root = build_range_tree(movies, dimensions)
    search_results_df = pd.DataFrame(search_range_tree(root, conditions))

    search_results_df['release_date'] = search_results_df['release_date'].apply(numeric_to_date)

    if genre_kw:  # perform lsh only if there is a genre keyword
        search_results_df['genre_names'] = search_results_df['genre_names'].apply(ast.literal_eval)  # cast to a list
        search_results_df['genre_names_cleaned'] = search_results_df['genre_names'].apply(lambda x: " ".join(x).lower())

        buckets = lsh.lsh(search_results_df)
        candidates = lsh.lsh_query(buckets, genre_kw)

        filtered_movies = search_results_df[search_results_df["id"].isin(candidates)].head(num_of_results)
        results = filtered_movies.values.tolist()

    else:
        results = search_results_df.values.tolist()

    return results
