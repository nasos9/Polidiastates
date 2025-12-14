import random

import mmh3
from collections import defaultdict

"""
Increase bands and decrease rows to have matches that are more exact -> Fewer candidates -> stricter results ->
                                                                       ->(documentary,music will not be found with keyword = documentary).
More candidates -> loose choice but -> more false positives and computation
"""

K = 3                      # shingle size
NUM_HASHES = 100          # number of minhash functions
P = 2**61 - 1             # large prime
HASH_FUNCS = []           # will be initialized once
RANDOM_SEED = 1093509     # fixed seed for reproducibility
NUM_BANDS = 25            # number of bands
NUM_ROWS = 4              # number of rows

def get_hash(s):
    return mmh3.hash(str(s)) & 0xffffffff  # 0xffffffff makes it unsigned (bitmask)

def init_hash_functions():
    global HASH_FUNCS
    random.seed(RANDOM_SEED)
    HASH_FUNCS = [(random.randint(1, P - 1), random.randint(0, P - 1), P) for _ in range(NUM_HASHES)]

def get_shingles(word):
    return {word[i:i+K] for i in range(len(word) - K + 1)}

def minhash_signature(shingles):
    sig = []
    for (a, b, p) in HASH_FUNCS:
        min_hash = min((a * get_hash(shingle) + b) % p for shingle in shingles)
        sig.append(min_hash)
    return tuple(sig)

def create_buckets(signatures):
    buckets = defaultdict(list)
    for key, signature in signatures.items():
        for b in range(NUM_BANDS):
            start = b * NUM_ROWS
            band = tuple(signature[start:start + NUM_ROWS])
            buckets[(b, get_hash(band))].append(key)

    return buckets

def lsh(dataframe):

    if not HASH_FUNCS:
        init_hash_functions()

    # SHINGLING
    shingle_sets = {}
    for index, row in dataframe.iterrows():
        shingles = get_shingles(row["genre_names_cleaned"])
        if shingles:
            shingle_sets[row["id"]] = shingles

    # MINHASH
    signatures = {
        movie_id: minhash_signature(shingle_set)
        for movie_id, shingle_set in shingle_sets.items()
    }

    # LSH
    buckets = create_buckets(signatures)
    return buckets

def lsh_query(buckets, keyword):

    keyword_shingles = get_shingles(keyword)
    keyword_signature = minhash_signature(keyword_shingles)
    candidates = set()

    for i in range(NUM_BANDS):
        start = i * NUM_ROWS
        band = tuple(keyword_signature[start:start + NUM_ROWS])
        bucket_key = (i, get_hash(band))

        if bucket_key in buckets:
            for movie_id in buckets[bucket_key]:
                candidates.add(movie_id)

    return candidates
