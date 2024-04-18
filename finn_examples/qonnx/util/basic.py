# Copyright (c) 2020 Xilinx, Inc.
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
#
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
#
# * Neither the name of Xilinx nor the names of its
#   contributors may be used to endorse or promote products derived from
#   this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import numpy as np
import os
import random
import string
import warnings
from qonnx.core.datatype import DataType

# TODO solve by moving onnx-dependent fxns to onnx.py
# finn-examples uses parts of qonnx without having
# onnx installed and doesn't use this functionality
# workaround to avoid import errors when onnx isn't
# installed:
try:
    from onnx.helper import make_model, make_opsetid
except ModuleNotFoundError:
    make_model = None
    make_opsetid = None


def get_preferred_onnx_opset():
    "Return preferred ONNX opset version for QONNX"
    return 11


def qonnx_make_model(graph_proto, **kwargs):
    "Wrapper around ONNX make_model with preferred qonnx opset version"
    opset_imports = kwargs.pop("opset_imports", None)
    if opset_imports is None:
        opset_imports = [make_opsetid("", get_preferred_onnx_opset())]
        kwargs["opset_imports"] = opset_imports
    else:
        kwargs["opset_imports"] = opset_imports
    return make_model(graph_proto, **kwargs)


def is_finn_op(op_type):
    "Return whether given op_type string is a QONNX or FINN custom op"
    return op_type.startswith("finn") or op_type.startswith("qonnx.custom_op")


def get_num_default_workers():
    """Return the number of workers for parallel transformations. Controllable
    via the NUM_DEFAULT_WORKERS environment variable. If the env.var. is
    undefined, the default value of 1 is returned.
    """

    try:
        return int(os.environ["NUM_DEFAULT_WORKERS"])
    except KeyError:
        return 1


def get_execution_error_thresh():
    "Return the max error that is allowed for rounding in QONNX execution."
    try:
        return float(os.environ["ERROR_THRESH"])
    except KeyError:
        return 1e-2


def get_sanitize_quant_tensors():
    """Return whether tensors with quantization annotations should be sanitized.
    Enabled by default, disabling will yield faster ONNX execution but may give
    incorrect results. Use with caution."""
    try:
        return int(os.environ["SANITIZE_QUANT_TENSORS"])
    except KeyError:
        # enabled by default
        return 1


def get_by_name(container, name, name_field="name"):
    """Return item from protobuf container by .name field if it exists, None otherwise.
    Will throw an Exception if multiple items are found, since this violates the
    ONNX standard."""
    names = [getattr(x, name_field) for x in container]

    inds = [i for i, e in enumerate(names) if e == name]
    if len(inds) > 1:
        raise Exception("Found multiple get_by_name matches, undefined behavior")
    elif len(inds) == 0:
        return None
    else:
        ind = inds[0]
        return container[ind]


def remove_by_name(container, name, name_field="name"):
    """Remove item from protobuf container by .name field if it exists."""
    item = get_by_name(container, name, name_field)
    if item is not None:
        container.remove(item)


def random_string(stringLength=6):
    """Randomly generate a string of letters and digits."""
    lettersAndDigits = string.ascii_letters + string.digits
    return "".join(random.choice(lettersAndDigits) for i in range(stringLength))


def interleave_matrix_outer_dim_from_partitions(matrix, n_partitions):
    """Interleave the outermost dimension of a matrix from given
    partitions (n_partitions)."""
    if type(matrix) != np.ndarray or matrix.dtype != np.float32:
        # try to convert to a float numpy array (container dtype is float)
        matrix = np.asarray(matrix, dtype=np.float32)
    shp = matrix.shape
    ndim = matrix.ndim
    # ensure # partitions evenly divide the outermost dimension
    assert (
        shp[0] % n_partitions == 0
    ), """The outermost dimension is not divisable
    by the number of partitions."""
    # only tested for matrices
    assert (
        ndim == 2
    ), """The dimension of the matrix is not 2. Currently this function
    only works for matrices."""
    # interleave rows between PEs using reshape + transpose
    matrix_r = matrix.reshape(-1, n_partitions, shp[1]).transpose((1, 0, 2))
    matrix_r = matrix_r.reshape(n_partitions, -1, shp[1])
    return matrix_r


def roundup_to_integer_multiple(x, factor):
    """Round up integer x to the nearest integer multiple of integer factor.
    Returns x if factor is set to -1. Both x and factor must otherwise be
    positive."""
    # ensure integers
    assert int(x) == x, "The input x is not an integer."
    assert int(factor) == factor, "The input factor is not an integer."
    # use -1 to indicate no padding needed
    if factor == -1:
        return x
    # ensure positive values
    assert factor > 0 and x > 0, "Factor and x are <= 0."
    if x < factor:
        return factor
    else:
        if x % factor == 0:
            return x
        else:
            return x + (factor - (x % factor))


def pad_tensor_to_multiple_of(ndarray, pad_to_dims, val=0, distr_pad=False):
    """Pad each dimension of given NumPy ndarray using val, so that each
    dimension is a multiple of the respective value in pad_to_dims. -1 means
    do not pad that particular dimension. If distr_pad is False, all padding
    will be inserted after the existing values; otherwise it will be split
    evenly between before and after the existing values, with one extra value
    inserted after if the padding amount is not divisible by two."""
    if type(ndarray) != np.ndarray or ndarray.dtype != np.float32:
        # try to convert to a float numpy array (container dtype is float)
        ndarray = np.asarray(ndarray, dtype=np.float32)
    assert ndarray.ndim == len(
        pad_to_dims
    ), """The dimensions of the input
    array don't match the length of the pad_to_dims value."""
    # compute the desired shape
    desired = zip(list(ndarray.shape), list(pad_to_dims))
    desired = map(lambda x: roundup_to_integer_multiple(x[0], x[1]), desired)
    desired = np.asarray(list(desired), dtype=np.int32)
    current = np.asarray(ndarray.shape, dtype=np.int32)
    pad_amt = desired - current
    # add padding to get to the desired shape
    if distr_pad:
        pad_before = (pad_amt // 2).astype(np.int32)
        pad_after = pad_amt - pad_before
        pad_amt = list(zip(pad_before, pad_after))
    else:
        # all padding is added after the existing values
        pad_amt = list(map(lambda x: (0, x), pad_amt))
    ret = np.pad(ndarray, pad_amt, mode="constant", constant_values=val)
    assert (
        np.asarray(ret.shape, dtype=np.int32) == desired
    ).all(), """The
    calculated output array doesn't match the desired/expected one."""
    return ret


def calculate_matvec_accumulator_range(matrix: np.ndarray, vec_dt: DataType):
    """Calculate the minimum and maximum possible result (accumulator) values
    for a dot product x * A, given matrix A of dims (MW, MH), and vector (1, MW)
    with datatype vec_dt. Returns (acc_min, acc_max).
    """
    max_weight = abs(matrix).sum(axis=0).max()
    max_input = max(abs(vec_dt.min()), abs(vec_dt.max()))
    max_value = max_input * max_weight
    # If either the weight and input datatypes are signed, then the minimum
    # value that their accumulated product can be is -max_value. Else, it's 0.
    min_value = -max_value if (matrix.min() < 0) or vec_dt.signed() else 0
    return (min_value, max_value)


def gen_finn_dt_tensor(finn_dt, tensor_shape):
    """Generates random tensor in given shape and with given QONNX DataType."""
    if type(tensor_shape) == list:
        tensor_shape = tuple(tensor_shape)
    if finn_dt == DataType["BIPOLAR"]:
        tensor_values = np.random.randint(2, size=tensor_shape)
        tensor_values = 2 * tensor_values - 1
    elif finn_dt == DataType["BINARY"]:
        tensor_values = np.random.randint(2, size=tensor_shape)
    elif "INT" in finn_dt.name or finn_dt == DataType["TERNARY"]:
        tensor_values = np.random.randint(finn_dt.min(), high=finn_dt.max() + 1, size=tensor_shape)
    elif "FIXED" in finn_dt.name:
        int_dt = DataType["INT" + str(finn_dt.bitwidth())]
        tensor_values = np.random.randint(int_dt.min(), high=int_dt.max() + 1, size=tensor_shape)
        tensor_values = tensor_values * finn_dt.scale_factor()
    elif finn_dt == DataType["FLOAT32"]:
        tensor_values = np.random.randn(*tensor_shape)
    else:
        raise ValueError(
            "Datatype {} is not supported, no tensor could be generated".format(finn_dt)
        )
    # always use float type as container
    return tensor_values.astype(np.float32)


def calculate_signed_dot_prod_range(dt_a, dt_b, len):
    """Returns the (min,max) values a dot product between two signed vectors of
    types dt_a and dt_b of len elements can take."""
    assert (
        dt_a.signed() and dt_b.signed()
    ), """The input values are not both
    signed vectors."""
    min_prod = 2**30
    max_prod = -(2**30)
    for a_val in [dt_a.min(), dt_a.max()]:
        for b_val in [dt_b.min(), dt_b.max()]:
            prod = a_val * b_val * len
            if prod < min_prod:
                min_prod = prod
            if prod > max_prod:
                max_prod = prod
    return (min_prod, max_prod)


def sanitize_quant_values(model, node_tensors, execution_context, check_values=False):
    """Sanitize given list of tensors in execution_context by rounding values
    that are supposed to be integers (as indicated by their quantization
    annotation). Will raise an assertion if the amount of rounding is too large.
    Returns the sanitized execution context.
    If check_values is specified, an extra DataType.allowed() check will be
    performed on any rounded tensors.
    Background:
    QONNX uses floating point tensors as a carrier data type to represent
    integers. Floating point arithmetic can introduce rounding errors, e.g.
    (int_num * float_scale) / float_scale is not always equal to int_num.
    We use this function to ensure that the values that are supposed to be
    integers are indeed integers.
    """

    for tensor_name in node_tensors:
        dtype = model.get_tensor_datatype(tensor_name)
        # floats don't need sanitization, skip to next
        # introduces less quicker runtime
        if dtype == DataType["FLOAT32"]:
            continue
        current_values = execution_context[tensor_name]
        updated_values = current_values
        has_to_be_rounded = False
        # TODO: vectorize with numpy
        for value in np.nditer(current_values):
            if not dtype.allowed(value):
                has_to_be_rounded = True
                break
        if has_to_be_rounded:
            updated_values = np.round(current_values)
            warnings.warn(
                "The values of tensor {} can't be represented "
                "with the set datatype annotation ({}), they will be rounded to match the "
                "datatype annotation.".format(tensor_name, dtype.name)
            )
        # check if rounded values are not too far from original values
        max_error = max(np.abs(current_values - updated_values).flatten())
        if max_error <= get_execution_error_thresh():
            if check_values is True:
                # check again if values can now be represented with set finn datatype
                # TODO: vectorize with numpy
                for value in np.nditer(updated_values):
                    if not dtype.allowed(value):
                        raise Exception(
                            """Values can't be represented with set
                                finn datatype ({}) for input {}""".format(
                                dtype, tensor_name
                            )
                        )
            execution_context[tensor_name] = updated_values
        else:
            raise Exception(
                """Rounding error is too high to match set QONNX
            datatype ({}) for input {}""".format(
                    dtype, tensor_name
                )
            )
    return
