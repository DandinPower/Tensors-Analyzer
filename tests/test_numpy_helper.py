import unittest
import numpy as np

import numpy_helper

class TestNumpyHelper(unittest.TestCase):

    def test_move_src_tensor_into_dst_tensor(self):
        dst = np.zeros(10, dtype=np.float32)
        src = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        dst_current_offset = 5
        
        new_offset = numpy_helper.move_src_tensor_into_dst_tensor(dst, src, dst_current_offset)
        
        self.assertEqual(new_offset, dst_current_offset + len(src))
        np.testing.assert_array_equal(dst[:dst_current_offset], np.zeros(dst_current_offset, dtype=np.float32))
        np.testing.assert_array_equal(dst[dst_current_offset:dst_current_offset + len(src)], src)
        np.testing.assert_array_equal(dst[dst_current_offset + len(src):], np.zeros(len(dst) - dst_current_offset - len(src), dtype=np.float32))

    def test_move_src_tensor_into_dst_tensor_insufficient_space(self):
        dst = np.zeros(5, dtype=np.float32)
        src = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        dst_current_offset = 3

        with self.assertRaises(RuntimeError):
            numpy_helper.move_src_tensor_into_dst_tensor(dst, src, dst_current_offset)

    def test_copy_src_tensor_into_dst_tensor(self):
        dst = np.zeros(5, dtype=np.float32)
        src = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32)

        numpy_helper.copy_src_tensor_into_dst_tensor(dst, src)
        
        np.testing.assert_array_equal(dst, src)

    def test_copy_src_tensor_into_dst_tensor_different_sizes(self):
        dst = np.zeros(5, dtype=np.float32)
        src = np.array([1.0, 2.0, 3.0], dtype=np.float32)

        with self.assertRaises(RuntimeError):
            numpy_helper.copy_src_tensor_into_dst_tensor(dst, src)

    def test_tensor_abs(self):
        tensor = np.array([-1.0, -2.0, 3.0, -4.0, 5.0], dtype=np.float32)
        
        numpy_helper.tensor_abs(tensor)
        
        expected_result = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32)
        np.testing.assert_array_equal(tensor, expected_result)

if __name__ == '__main__':
    unittest.main()
