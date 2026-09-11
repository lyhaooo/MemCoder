from solution import merge_intervals

assert merge_intervals([]) == []
assert merge_intervals([[1, 3]]) == [[1, 3]]
assert merge_intervals([[1, 3], [2, 6], [8, 10], [15, 18]]) == [
    [1, 6],
    [8, 10],
    [15, 18],
]
assert merge_intervals([[1, 4], [4, 5]]) == [[1, 5]]

