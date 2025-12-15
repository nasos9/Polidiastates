import ast

import pandas as pd
from datetime import datetime
import lsh
import numpy as np
import time

# ======== PREPROCESSING HELPERS ==========

def convert_date_to_numeric(date_str):
    """Convert date string to numeric format YYYYMMDD"""
    try:
        # Αν είναι ήδη αριθμός/float (π.χ. από το Excel), το επιστρέφουμε ως int
        if isinstance(date_str, (int, float)):
            return int(date_str)
        return int(datetime.strptime(str(date_str), "%Y-%m-%d").strftime("%Y%m%d"))
    except:
        return 0  # Default για λάθη

def log_normalize_get_stats(series):
    """Log normalization returning the series AND the min/max used"""
    series = series.fillna(series.median())
    series = series.replace(0, 1)  # Avoid log(0)
    log_vals = np.log1p(series)
    min_val = log_vals.min()
    max_val = log_vals.max()

    if max_val - min_val == 0:
        return (series * 0), 0, 1

    norm_series = (log_vals - min_val) / (max_val - min_val)
    return norm_series, min_val, max_val

def minmax_get_stats(series):
    """Min-max normalization returning the series AND the min/max used"""
    series = series.fillna(series.median())
    min_val = series.min()
    max_val = series.max()

    if max_val - min_val == 0:
        return (series * 0), 0, 1

    norm_series = (series - min_val) / (max_val - min_val)
    return norm_series, min_val, max_val

# ========== OCTREE NODE (ΑΜΕΤΑΒΛΗΤΟ) ===========
class OctreeNode:
    def __init__(self, bounds, capacity=8, max_depth=20, current_depth=0):
        self.bounds = bounds
        self.capacity = capacity
        self.points = []
        self.children = None
        self.max_depth = max_depth
        self.current_depth = current_depth

    def is_within_bounds(self, point):
        for i in range(3):
            if not (self.bounds[i][0] <= point[i] <= self.bounds[i][1]):
                return False
        return True

    def split(self):
        if self.current_depth >= self.max_depth:
            return

        x_mid = (self.bounds[0][0] + self.bounds[0][1]) / 2
        y_mid = (self.bounds[1][0] + self.bounds[1][1]) / 2
        z_mid = (self.bounds[2][0] + self.bounds[2][1]) / 2

        sub_bounds = [
            [[self.bounds[0][0], x_mid], [self.bounds[1][0], y_mid], [self.bounds[2][0], z_mid]],
            [[x_mid, self.bounds[0][1]], [self.bounds[1][0], y_mid], [self.bounds[2][0], z_mid]],
            [[self.bounds[0][0], x_mid], [y_mid, self.bounds[1][1]], [self.bounds[2][0], z_mid]],
            [[x_mid, self.bounds[0][1]], [y_mid, self.bounds[1][1]], [self.bounds[2][0], z_mid]],
            [[self.bounds[0][0], x_mid], [self.bounds[1][0], y_mid], [z_mid, self.bounds[2][1]]],
            [[x_mid, self.bounds[0][1]], [self.bounds[1][0], y_mid], [z_mid, self.bounds[2][1]]],
            [[self.bounds[0][0], x_mid], [y_mid, self.bounds[1][1]], [z_mid, self.bounds[2][1]]],
            [[x_mid, self.bounds[0][1]], [y_mid, self.bounds[1][1]], [z_mid, self.bounds[2][1]]],
        ]

        self.children = [OctreeNode(b, self.capacity, self.max_depth, self.current_depth + 1)
                         for b in sub_bounds]

    def insert(self, point, full_data):
        if not self.is_within_bounds(point):
            return False

        if self.children is None:
            if len(self.points) < self.capacity or self.current_depth >= self.max_depth:
                self.points.append((point, full_data))
                return True
            else:
                self.split()
                if self.children is None:
                    self.points.append((point, full_data))
                    return True
                for p, d in self.points:
                    inserted = False
                    for child in self.children:
                        if child.insert(p, d):
                            inserted = True
                            break
                    if not inserted:
                        continue
                self.points = []

        for child in self.children:
            if child.insert(point, full_data):
                return True

        return False

    def range_query(self, range_min, range_max, results=None):
        if results is None:
            results = []

        for i in range(3):
            if self.bounds[i][1] < range_min[i] or self.bounds[i][0] > range_max[i]:
                return results

        for point, full_data in self.points:
            if all(range_min[i] <= point[i] <= range_max[i] for i in range(3)):
                results.append(full_data)

        if self.children is not None:
            for child in self.children:
                child.range_query(range_min, range_max, results)

        return results

# ========== MAIN OCTREE + LSH ==========
def movies_octree_main(conditions, genre_keywords, num_of_results):
    """
    Main function.
    :param conditions: Dict με πραγματικές τιμές!
                       π.χ. 'budget': (1000000, 50000000)
                            'release_date': ('1990-01-01', '2005-12-31')
    """
    if conditions is None:
        conditions = {}

    print("Διάβασμα δεδομένων...")
    df = pd.read_csv("movies_testing.csv")

    print("Προεπεξεργασία και Κανονικοποίηση...")

    # 1. Date Conversion
    df['release_date_numeric'] = df['release_date'].apply(convert_date_to_numeric)

    # 2. Normalization με αποθήκευση των ορίων (Min/Max)
    # Budget: Log Normalization
    df['budget_norm'], log_b_min, log_b_max = log_normalize_get_stats(df['budget'])

    # Popularity: MinMax
    df['popularity_norm'], pop_min, pop_max = minmax_get_stats(df['popularity'])

    # Release Date: MinMax
    df['release_norm'], date_min, date_max = minmax_get_stats(df['release_date_numeric'])

    # Jitter to avoid duplicates
    np.random.seed(42)
    jitter = 1e-9
    df['budget_norm'] += np.random.uniform(-jitter, jitter, len(df))
    df['popularity_norm'] += np.random.uniform(-jitter, jitter, len(df))
    df['release_norm'] += np.random.uniform(-jitter, jitter, len(df))

    # Build Octree
    columns_for_splitting = ['budget_norm', 'popularity_norm', 'release_norm']
    points = df[columns_for_splitting].values.tolist()
    full_data = df.values.tolist()

    bounds = [
        [df['budget_norm'].min(), df['budget_norm'].max()],
        [df['popularity_norm'].min(), df['popularity_norm'].max()],
        [df['release_norm'].min(), df['release_norm'].max()],
    ]

    print("Δημιουργία Octree...")
    octree = OctreeNode(bounds, capacity=50, max_depth=15)
    for i, (point, row) in enumerate(zip(points, full_data)):
        if i % 10000 == 0 and i > 0:
            print(f"  Εισαγωγή {i}...")
        octree.insert(point, row)

    # ============ QUERY TRANSFORMATION ============
    # Εδώ μετατρέπουμε τις ΠΡΑΓΜΑΤΙΚΕΣ τιμές του χρήστη σε NORMALIZED τιμές για το δέντρο

    # Ορίζουμε ένα μικρό epsilon, λίγο μεγαλύτερο από το jitter που έβαλες (1e-9)
    epsilon = 1e-8

    # 1. Budget Transform
    raw_budget_range = conditions.get('budget', (-1, float('inf')))
    b_min_q = raw_budget_range[0] if raw_budget_range[0] > 0 else 1
    b_max_q = raw_budget_range[1]
    q_budget_min_norm = (np.log1p(b_min_q) - log_b_min) / (log_b_max - log_b_min)
    q_budget_max_norm = (np.log1p(b_max_q) - log_b_min) / (log_b_max - log_b_min)

    # --- ΔΙΟΡΘΩΣΗ: Επέκταση ορίων ---
    q_budget_min_norm -= epsilon
    q_budget_max_norm += epsilon

    # 2. Popularity Transform
    raw_pop_range = conditions.get('popularity', (0, float('inf')))
    q_pop_min_norm = (raw_pop_range[0] - pop_min) / (pop_max - pop_min)
    q_pop_max_norm = (raw_pop_range[1] - pop_min) / (pop_max - pop_min)

    # --- ΔΙΟΡΘΩΣΗ ---
    q_pop_min_norm -= epsilon
    q_pop_max_norm += epsilon

    # 3. Date Transform
    raw_date_range = conditions.get('release_date', ('1900-01-01', '2100-01-01'))
    d_min_int = convert_date_to_numeric(raw_date_range[0])
    d_max_int = convert_date_to_numeric(raw_date_range[1])
    q_date_min_norm = (d_min_int - date_min) / (date_max - date_min)
    q_date_max_norm = (d_max_int - date_min) / (date_max - date_min)

    # --- ΔΙΟΡΘΩΣΗ ---
    q_date_min_norm -= epsilon
    q_date_max_norm += epsilon

    # Φτιάχνουμε τα τελικά όρια για το δέντρο
    range_min = [q_budget_min_norm, q_pop_min_norm, q_date_min_norm]
    range_max = [q_budget_max_norm, q_pop_max_norm, q_date_max_norm]

    # Εκτέλεση αναζήτησης
    print(f"Αναζήτηση Range (Normalized): {range_min} έως {range_max}")
    start_time = time.time()
    search_results = octree.range_query(range_min, range_max)
    end_time = time.time()
    print(f"Βρέθηκαν {len(search_results)} ταινίες.")
    print(f"Χρόνος αναζήτησης Octree: {end_time - start_time:.6f} δευτερόλεπτα.")

    columns = ["id", "title", "adult", "original_language", "origin_country", "release_date", "genre_names", "production_company_names", "budget", "revenue", "runtime", "popularity", "vote_average", "vote_count"]
    query_results_dict = [dict(zip(columns, row)) for row in search_results]
    search_results_df = pd.DataFrame(query_results_dict)

    # LSH Logic
    if genre_keywords:  # perform lsh only if there is a genre keyword
        search_results_df['genre_names'] = search_results_df['genre_names'].apply(ast.literal_eval)  # cast to a list
        search_results_df['genre_names_cleaned'] = search_results_df['genre_names'].apply(lambda x: " ".join(x).lower())

        buckets = lsh.lsh(search_results_df)
        candidates = lsh.lsh_query(buckets, genre_keywords)

        filtered_movies = search_results_df[search_results_df["id"].isin(candidates)].head(num_of_results)
        results = filtered_movies.values.tolist()

    else:
        results = search_results_df.values.tolist()

    return results
