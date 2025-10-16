def grid_for_rmsnorm_linear_layer(size):
    # 96 and 64 are enough to cover all Qwen3 model? Please update the method
    # if you meet any incompatibility.
    if size % 96 == 0:
        return 96
    elif size % 64 == 0:
        return 64
    
# Return the largest factor of m that is less than or equal to n
# This is used to determine the grid size
def max_factor_leq_n(m: int, n: int) -> int:
    max_factor = 1
    i = 1
    while i * i <= m:
        if m % i == 0:
            if i <= n:
                max_factor = max(max_factor, i)
            if m // i <= n:
                max_factor = max(max_factor, m // i)
        i += 1
    return max_factor