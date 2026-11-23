"""Shared reference helpers for the SRS resource-mapping regression tests."""


def _reference_low_papr_phase(sequence_length, group_index, base_sequence_index, n):
    zc_length = max(
        candidate
        for candidate in range(sequence_length - 1, 1, -1)
        if all(candidate % divisor for divisor in range(2, int(candidate**0.5) + 1))
    )
    q_numerator = zc_length * (group_index + 1)
    q_bar = (2 * q_numerator + 31) // 62
    q_sign = (-1) ** ((2 * q_numerator) // 31)
    q = q_bar + base_sequence_index * q_sign
    return q * (n % zc_length) * (n % zc_length + 1) / zc_length


def _scalar_gold_bits(c_init: int, count: int) -> list[int]:
    x1 = [1] + [0] * (1600 + count + 30)
    x2 = [(c_init >> bit) & 1 for bit in range(31)] + [0] * (1600 + count)
    for index in range(1600 + count):
        x1[index + 31] = x1[index + 3] ^ x1[index]
        x2[index + 31] = x2[index + 3] ^ x2[index + 2] ^ x2[index + 1] ^ x2[index]
    return [x1[1600 + index] ^ x2[1600 + index] for index in range(count)]
