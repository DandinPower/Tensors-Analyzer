#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <half.hpp>
#include <omp.h>

using half = half_float::half;
namespace py = pybind11;

/**
 * Moves the content of the source tensor into the destination tensor.
 * 
 * @param dst The destination tensor where the content of the source tensor will be moved. The dst tensor is assumed to have enough space to store the content of the src tensor.
 * @param src The source tensor whose content will be moved into the destination tensor.
 * @param dst_current_offset The current offset in the destination tensor where the content of the source tensor will be moved.
 * @return The updated offset in the destination tensor after moving the content of the source tensor.
 */
size_t move_src_tensor_into_dst_tensor(py::array_t<float> dst, py::array_t<float> src, size_t dst_current_offset) {
    py::buffer_info src_buf = src.request(), dst_buf = dst.request();
    int src_size = src_buf.size;
    int dst_size = dst_buf.size;
    
    if (dst_current_offset + src_size > dst_size) {
        throw std::runtime_error("Destination tensor does not have enough space to store the source tensor");
    }

    auto src_ptr = static_cast<float *>(src_buf.ptr);
    auto dst_ptr = static_cast<float *>(dst_buf.ptr);

    memcpy(dst_ptr + dst_current_offset, src_ptr, src_size * sizeof(float));
    return dst_current_offset + src_size;
}

/**
 * Copies the contents of the source tensor into the destination tensor.
 * 
 * @param dst The destination tensor to copy into.
 * @param src The source tensor to copy from.
 * 
 * @throws std::runtime_error if the input tensors have different sizes.
 */
void copy_src_tensor_into_dst_tensor(py::array_t<float> dst, py::array_t<float> src) {
    py::buffer_info src_buf = src.request(), dst_buf = dst.request();
    int src_size = src_buf.size;
    int dst_size = dst_buf.size;
    
    if (src_size != dst_size) {
        throw std::runtime_error("Input tensors must have the same size");
    }

    auto src_ptr = static_cast<float *>(src_buf.ptr);
    auto dst_ptr = static_cast<float *>(dst_buf.ptr);

    memcpy(dst_ptr, src_ptr, src_size * sizeof(float));
}

/**
 * Takes the absolute value of each element in the input tensor.
 *
 * @param tensor - The input tensor.
 */
void tensor_abs(py::array_t<float> tensor) {
    py::buffer_info tensor_buf = tensor.request();

    auto tensor_ptr = static_cast<float *>(tensor_buf.ptr);

#pragma omp parallel for
    for (size_t i = 0; i < tensor_buf.size; i++) {
        tensor_ptr[i] = std::abs(tensor_ptr[i]);
    }
}

PYBIND11_MODULE(numpy_helper, m) {
    m.def("move_src_tensor_into_dst_tensor", &move_src_tensor_into_dst_tensor, "A function which moves the content of the source tensor into the destination tensor");
    m.def("copy_src_tensor_into_dst_tensor", &copy_src_tensor_into_dst_tensor, "A function which copies the content of the source tensor into the destination tensor");
    m.def("tensor_abs", &tensor_abs, "A function which computes the absolute value of a tensor");
}