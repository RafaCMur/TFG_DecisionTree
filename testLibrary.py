import numpy as np

# Counts the number of 'num' in any array (even 2D arrays)
def count(array, num, tolerance=1e-6):
    count = np.sum(np.isclose(array, num, atol=tolerance))
    return count

# Test of count_ones()
def test_count():
    arr_2d = np.array([[0, 1, 0.4, 1, 0.6],
                   [0.2, 1, 0.5, 0.2, 1],
                   [0.3, 0.2, 0.5, 0.2, 0.1],
                   [0.5, 0.8, 1, 0.3, 0.4]])

    ones_count = count(arr_2d, 1)
    print("Number of values equal to 1 or approximately 1.0:", ones_count)

# See if there are repeated elements in an array
def has_repeated_elements(array):
    return len(array) != len(set(array))

# See if there are common elements in two arrays
def has_common_elements(array1, array2):
    return len(set(array1).intersection(array2)) > 0

# See if there are uncommon elements in two arrays even if they are unordered
def has_uncommon_elements(array1, array2):
    return len(set(array1).symmetric_difference(array2)) > 0

# Count uncommon elements in two arrays
def count_uncommon_elements(array1, array2):
    return len(set(array1).symmetric_difference(array2))

# Returns if subset is a subset of superset
def is_subset(subset, superset):
    return set(subset).issubset(set(superset))

# Test of is_subset()
def test_is_subset():
    superset = set([1, 2, 3, 4, 5])
    subset = set([1, 2, 3])
    print("Is subset a subset of superset? ", is_subset(subset, superset))
    superset = set([1, 2, 3, 4, 5])
    subset = set([1, 2, 3, 6])
    print("Is subset a subset of superset? ", is_subset(subset, superset))

def test_transform_to_binary(ds):
    print("Number of 0's in y_train", count(ds.y_train, 0))
    print("Number of 1's in y_train", count(ds.y_train, 1))