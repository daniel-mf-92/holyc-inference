#!/usr/bin/env python3
import unittest

FNV_OFFSET = 1469598103934665603
FNV_PRIME = 1099511628211
MASK64 = (1 << 64) - 1
MASK63 = (1 << 63) - 1

PREFIX_CACHE_OK = 0
PREFIX_CACHE_ERR_NULL_PTR = 1
PREFIX_CACHE_ERR_BAD_PARAM = 2


def holyc_like_prefix_hash_checked(tokens, token_count, out_is_null=False, tokens_is_null=False):
    if tokens_is_null or out_is_null:
        return PREFIX_CACHE_ERR_NULL_PTR, None
    if token_count < 0:
        return PREFIX_CACHE_ERR_BAD_PARAM, None

    h = FNV_OFFSET
    idx = 0
    while idx < token_count:
        v = tokens[idx]
        h = (h ^ v) & MASK64
        h = (h * FNV_PRIME) & MASK64
        idx += 1

    h = (h ^ token_count) & MASK64
    h = (h * FNV_PRIME) & MASK64
    h = h & MASK63
    return PREFIX_CACHE_OK, h


class PrefixHashQ64CheckedTests(unittest.TestCase):
    def test_null_ptr_guard(self):
        st, out = holyc_like_prefix_hash_checked([1, 2], 2, out_is_null=True)
        self.assertEqual(st, PREFIX_CACHE_ERR_NULL_PTR)
        self.assertIsNone(out)

        st, out = holyc_like_prefix_hash_checked(None, 2, tokens_is_null=True)
        self.assertEqual(st, PREFIX_CACHE_ERR_NULL_PTR)
        self.assertIsNone(out)

    def test_negative_count_guard(self):
        st, out = holyc_like_prefix_hash_checked([], -1)
        self.assertEqual(st, PREFIX_CACHE_ERR_BAD_PARAM)
        self.assertIsNone(out)

    def test_known_vectors(self):
        vectors = [
            [],
            [0],
            [1],
            [255],
            [1, 2, 3, 4],
            [255, 0, 255, 0, 1, 2, 3],
            list(range(32)),
        ]
        expected = []
        for toks in vectors:
            st, h = holyc_like_prefix_hash_checked(toks, len(toks))
            self.assertEqual(st, PREFIX_CACHE_OK)
            expected.append(h)

        self.assertEqual(expected[0], 4953163356653287321)
        self.assertEqual(expected[1], 1903072112059922248)
        self.assertEqual(expected[2], 1902117735966824325)
        self.assertEqual(expected[3], 1896378285268698255)
        self.assertEqual(expected[4], 4733238719661419597)
        self.assertEqual(expected[5], 6138617849566852678)
        self.assertEqual(expected[6], 2689684201023061881)

    def test_count_domain_separation(self):
        toks = [7, 8, 9, 10]
        st_a, h_a = holyc_like_prefix_hash_checked(toks, 4)
        st_b, h_b = holyc_like_prefix_hash_checked(toks + [0], 5)
        self.assertEqual(st_a, PREFIX_CACHE_OK)
        self.assertEqual(st_b, PREFIX_CACHE_OK)
        self.assertNotEqual(h_a, h_b)

    def test_high_bit_cleared(self):
        for n in [0, 1, 2, 7, 32, 128, 255]:
            toks = [(i * 13 + 5) & 0xFF for i in range(n)]
            st, h = holyc_like_prefix_hash_checked(toks, len(toks))
            self.assertEqual(st, PREFIX_CACHE_OK)
            self.assertGreaterEqual(h, 0)
            self.assertLessEqual(h, MASK63)


if __name__ == "__main__":
    unittest.main()
