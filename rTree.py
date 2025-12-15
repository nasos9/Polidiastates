import pandas as pd
import time
from datetime import datetime
import lsh
import argparse
import ast


class MBR:
    def __init__(self, mins, maxs):
        self.dim = len(mins)  # dimensions
        self.mins = mins  # list of minimum values per dimensions
        self.maxs = maxs  # list of maximum values per dimensions

    def area(self):
        total = 1.0
        # area = product of edge_length per dimensions
        # e.g. 2d -> area = length_x * length_y
        for i in range(self.dim):
            edge_length = self.maxs[i] - self.mins[i]
            total *= edge_length
        return total

    def intersects(self, other):
        # in any dimention
        for i in range(self.dim):
            # if other MBR is outside this MBR
            if self.maxs[i] < other.mins[i] or self.mins[i] > other.maxs[i]:
                return False
        return True

    def union(self, other):
        new_mins = []
        new_maxs = []
        # expand current MBR to include new MBR
        for i in range(self.dim):
            new_mins.append(min(self.mins[i], other.mins[i]))
            new_maxs.append(max(self.maxs[i], other.maxs[i]))
        return MBR(new_mins, new_maxs)

    def enlargement(self, other):
        # calculates the expansion needed in order for the current MBR to include the new MBR
        union_rect = self.union(other)
        return union_rect.area() - self.area()


class RtreeNode:
    def __init__(self, parent, is_leaf=False):
        self.parent = parent
        self.is_leaf = is_leaf
        # entries list:
        # leaf         : contains tuples (MBR, data_object)
        # internal node: contains tuples (MBR, RtreeNode)
        self.entries = []
        self.mbr: MBR = None

    def update_mbr(self):
        # calculates entire node MBR based on every entry
        if not self.entries:
            self.mbr = None
            return

        # get first MBR
        new_mbr = self.entries[0][0]
        # add all others
        for i in range(1, len(self.entries)):
            new_mbr = new_mbr.union(self.entries[i][0])
        self.mbr = new_mbr

    def update_mbr_for_insert(self, data_mbr):
        # calculates entire node MBR based on new entry
        new_mbr = None

        if self.mbr is not None:
            # node already has MBR so just include new insertion
            new_mbr = self.mbr.union(data_mbr)
        else:
            # first insertion to node so node_MBR = data_mbr
            new_mbr = data_mbr

        self.mbr = new_mbr


class Rtree():
    def __init__(self, m: int, M: int, dim: int):
        self.m = m
        self.M = M
        self.dim = dim
        self.root = RtreeNode(parent=None, is_leaf=True)

    def insert(self, record_vector, record_data):
        # record_vector: list of values for every dimention
        # record_data: the full record

        if len(record_vector) != self.dim:
            raise ValueError(f"Vector dimension: {len(record_vector)} doesnt match tree dimension: {self.dim}")

        # the MBR of a single record is a point
        data_mbr = MBR(record_vector, record_vector)

        # choose leaf
        leaf: RtreeNode = self.choose_leaf(self.root, data_mbr)

        # add entry
        leaf.entries.append((data_mbr, record_data))

        # update leaf MBR
        old_leaf_mbr = leaf.mbr
        leaf.update_mbr()

        if len(leaf.entries) > self.M:
            # node full so split
            self.split(leaf)
        else:
            if old_leaf_mbr != leaf.mbr:
                # if leaf MBR changed, update parent node's MBR
                self.adjust_tree(leaf)

    def adjust_tree(self, node: RtreeNode):
        if node.parent is None:
            # node == root
            return

        parent: RtreeNode = node.parent

        # find the entry in parent that points to node
        # update MBR for that entry
        for i, (stored_mbr, child_node) in enumerate(parent.entries):
            if child_node == node:
                # update stored_mbr for 'node' in parent with updated node.mbr
                parent.entries[i] = (node.mbr, node)

                # update entire MBR of parent
                parent.update_mbr()

                # recursively go up
                self.adjust_tree(parent)
                return

    def choose_leaf(self, node: RtreeNode, data_mbr):
        # find leaf that needs least enlargement

        if node.is_leaf:
            return node

        best_child: RtreeNode = None
        min_enlargement = float('inf')

        for child_mbr, child_node in node.entries:
            # for each entry in a node calculate enlargement needed to include new insertion
            enlargement = child_mbr.enlargement(data_mbr)
            if enlargement < min_enlargement:
                # found new minimum enlargement so replace old
                min_enlargement = enlargement
                best_child = child_node
            elif enlargement == min_enlargement:
                # found tie so resolve with smallest area
                if child_mbr.area() < best_child.mbr.area():
                    best_child = child_node

        return self.choose_leaf(best_child, data_mbr)

    def split(self, current_node: RtreeNode):
        # quadratic split

        # new sibling current_node
        new_node = RtreeNode(parent=current_node.parent, is_leaf=current_node.is_leaf)

        # pick the worst two entries to put together
        idx1, idx2 = self.pick_entries(current_node.entries)

        entry1 = current_node.entries[idx1]
        entry2 = current_node.entries[idx2]

        # entries of current and new nodes
        current_node_group = [entry1]
        new_node_group = [entry2]

        # current and new nodes MBRs
        current_node_mbr = entry1[0]
        new_node_mbr = entry2[0]

        # all other entries
        for i, entry in enumerate(current_node.entries):
            if i == idx1 or i == idx2:
                continue

            entry_mbr = entry[0]

            # check smallest group enlargement
            current_node_enlargement = current_node_mbr.enlargement(entry_mbr)
            new_node_enlargement = new_node_mbr.enlargement(entry_mbr)

            # add entry to that group
            if current_node_enlargement < new_node_enlargement:
                current_node_group.append(entry)
                current_node_mbr = current_node_mbr.union(entry_mbr)
            else:
                new_node_group.append(entry)
                new_node_mbr = new_node_mbr.union(entry_mbr)

        # change actual nodes
        current_node.entries = current_node_group
        current_node.mbr = current_node_mbr

        new_node.entries = new_node_group
        new_node.mbr = new_node_mbr

        # handle parents

        # update parent of moved entries
        if not current_node.is_leaf:
            for _, child_node in new_node.entries:
                child_node.parent = new_node

        if current_node.parent is None:
            # current_node is the root
            new_root = RtreeNode(parent=None, is_leaf=False)

            # assign parent of current and new nodes 
            current_node.parent = new_root
            new_node.parent = new_root

            # add the nodes
            new_root.entries.append((current_node.mbr, current_node))
            new_root.entries.append((new_node.mbr, new_node))

            # update root MBR
            new_root.update_mbr()

            self.root = new_root
        else:
            parent = current_node.parent

            # update current_node MBR in parent
            for i, (stored_mbr, child) in enumerate(parent.entries):
                if child == current_node:
                    parent.entries[i] = (current_node.mbr, current_node)
                    break

            # add new sibling node
            parent.entries.append((new_node.mbr, new_node))

            # update parent MBR
            parent.update_mbr()

            # check if parent overflows
            if len(parent.entries) > self.M:
                self.split(parent)
            else:
                # update all parents MBRs
                self.adjust_tree(parent)

    def pick_entries(self, entries):
        # find two entries that if put in the same MBR, create the most dead space.
        max_waste = -1
        seed1_idx, seed2_idx = 0, 1

        # compare every pair
        for i in range(len(entries)):
            for j in range(i + 1, len(entries)):
                mbr1 = entries[i][0]
                mbr2 = entries[j][0]

                total_mbr = mbr1.union(mbr2)

                waste = total_mbr.area() - mbr1.area() - mbr2.area()

                if waste > max_waste:
                    max_waste = waste
                    seed1_idx, seed2_idx = i, j

        return seed1_idx, seed2_idx

    def search(self, search_mbr):
        return self.search_recursive(self.root, search_mbr)

    def search_recursive(self, node, search_mbr):
        results = []

        # for every entry in current node
        for entry_mbr, entry_obj in node.entries:

            # skip entries that don't intersect with search area
            if not search_mbr.intersects(entry_mbr):
                continue

            if node.is_leaf:
                # node == leaf so entry_obj is actual data
                results.append(entry_obj)
            else:
                # node == internal so entry_obj is a child Node
                child_results = self.search_recursive(entry_obj, search_mbr)
                results.extend(child_results)

        return results


########################################################################################################
########################################################################################################
########################################################################################################
########################################################################################################


# utility
parser = argparse.ArgumentParser()
parser.add_argument('--verbose', '-v', action='store_true', help="Print search results")
parser.add_argument('--rows', type=int, help="Number of dataset rows to use")
args = parser.parse_args()

VERBOSE = args.verbose
NROWS = None  # args.rows


def read_csv_data(filename: str, nrows: int) -> pd.DataFrame:
    try:
        print("Reading csv Data")
        # if nrows has been given then set it else set to None(read all rows)
        return pd.read_csv(filename, nrows=(nrows if not None else None))
    except Exception as e:
        print(f"Error loading file: {e}")


def construct_rtree(data: pd.DataFrame, dimentions: int) -> Rtree:
    tree = Rtree(m=2, M=5, dim=dimentions)

    for _, row in data.iterrows():
        # skip NaNs
        if pd.isna(row['popularity']) or pd.isna(row['release_date']) or pd.isna(row['budget']):
            continue

        # create the vector/point
        vector = [
            float(row['popularity']),
            float(row['release_date']),
            float(row['budget'])
        ]

        # insert (vector, full row data)
        tree.insert(vector, row.to_dict())

    return tree


def convert_date_to_numeric(date_str):
    """Convert date string to numeric format YYYYMMDD"""
    try:
        # Αν είναι ήδη αριθμός/float (π.χ. από το Excel), το επιστρέφουμε ως int
        if isinstance(date_str, (int, float)):
            return int(date_str)
        return int(datetime.strptime(str(date_str), "%Y-%m-%d").strftime("%Y%m%d"))
    except:
        return 0  # Default για λάθη


def numeric_to_date(numeric_date):
    s = str(numeric_date)  # convert to string
    year = s[:4]
    month = s[4:6]
    day = s[6:8]
    return f"{year}-{month}-{day}"


def linear_search(data: pd.DataFrame, min_values, max_values):
    res = []
    for _, row in data.iterrows():
        if (min_values[0] <= row['popularity'] <= max_values[0]) and (
                min_values[1] <= row['release_date'] <= max_values[1]) and (
                min_values[2] <= row['budget'] <= max_values[2]):
            res.append(row.to_dict())
    return res


def validate_results(data, search_mins, search_maxs, results):
    linear_results = linear_search(data, search_mins, search_maxs)
    print(f"Found {len(linear_results)} matches")

    if len(results) != len(linear_results):
        print('Wrong results!')
        return

    results.sort(key=lambda x: x['id'])
    linear_results.sort(key=lambda x: x['id'])

    if linear_results == results:
        print('Rtree search results are correct!')
    else:
        print('Wrong results!')


def r_tree_main(conditions: dict, genre_kw, num_of_results):
    start_time = time.time()

    # read and format(date) data
    data = read_csv_data('movies_testing.csv', NROWS)
    data['release_date'] = data['release_date'].apply(convert_date_to_numeric)

    if VERBOSE:
        print("\n--- DATA STATISTICS ---")
        print(data[['popularity', 'release_date', 'budget']].describe())
        print("-----------------------")

    read_time = time.time()
    print(f"Done! Reading and formatting data took: {read_time - start_time:.2f}s")

    # insert data / construct tree
    print(f"\nInserting {len(data)} records into R-tree")
    rtree = construct_rtree(data, len(conditions))
    construct_time = time.time()
    print(f"Done! Inserting data took: {construct_time - read_time:.2f}s")

    # search
    # popularity  :  mins
    # release date:  yyyy-mm-dd
    # budget      :  usd

    search_mins = [conditions['popularity'][0], convert_date_to_numeric(conditions['release_date'][0]),
                   conditions['budget'][0]]
    search_maxs = [conditions['popularity'][1], convert_date_to_numeric(conditions['release_date'][1]),
                   conditions['budget'][1]]
    search_rect = MBR(search_mins, search_maxs)

    print(
        f"\nSearching for popularity=[{search_mins[0]}, {search_maxs[0]}], release_date=[{datetime.strptime(str(search_mins[1]), "%Y%m%d").strftime("%Y-%m-%d")}, {datetime.strptime(str(search_maxs[1]), "%Y%m%d").strftime("%Y-%m-%d")}], budget=[{search_mins[2]}, {search_maxs[2]}]")
    results = pd.DataFrame(rtree.search(search_rect))
    search_time = time.time()
    print(f"Done! Searching took: {search_time - construct_time:.6f}s")
    print(f"Found {len(results)} matches")
    if VERBOSE:
        for res in results:
            print(
                f" - {res['title']} (popularity: {res['popularity']}, release_date: {datetime.strptime(str(res['release_date']), "%Y%m%d").strftime("%Y-%m-%d")}, budget: {res['budget']}")

    # print('\nValidating results')
    # validate_results(data, search_mins, search_maxs, results)

    results['release_date'] = results['release_date'].apply(numeric_to_date)

    # perform lsh only if there is a genre keyword
    if genre_kw:
        results['genre_names'] = results['genre_names'].apply(ast.literal_eval)  # cast to a list
        results['genre_names_cleaned'] = results['genre_names'].apply(lambda x: " ".join(x).lower())

        buckets = lsh.lsh(results)
        candidates = lsh.lsh_query(buckets, genre_kw)

        filtered_movies = results[results["id"].isin(candidates)].head(num_of_results)
        results = filtered_movies.values.tolist()
        return results
    else:
        return results.values.tolist()
