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
from abc import ABC, abstractmethod
from enum import Enum, EnumMeta


class BaseDataType(ABC):
    "Base class for QONNX data types."

    def signed(self):
        "Returns whether this DataType can represent negative numbers."
        return self.min() < 0

    def __eq__(self, other):
        if isinstance(other, BaseDataType):
            return self.get_canonical_name() == other.get_canonical_name()
        elif isinstance(other, str):
            return self.get_canonical_name() == other
        else:
            return NotImplemented

    def __hash__(self):
        return hash(self.get_canonical_name())

    @property
    def name(self):
        return self.get_canonical_name()

    def __repr__(self):
        return self.get_canonical_name()

    def __str__(self):
        return self.get_canonical_name()

    @abstractmethod
    def bitwidth(self):
        "Returns the number of bits required for this DataType."
        pass

    @abstractmethod
    def min(self):
        "Returns the smallest possible value allowed by this DataType."
        pass

    @abstractmethod
    def max(self):
        "Returns the largest possible value allowed by this DataType."
        pass

    @abstractmethod
    def allowed(self, value):
        """Check whether given value is allowed for this DataType.

        * value (float32): value to be checked"""
        pass

    @abstractmethod
    def get_num_possible_values(self):
        """Returns the number of possible values this DataType can take. Only
        implemented for integer types for now."""
        pass

    @abstractmethod
    def is_integer(self):
        "Returns whether this DataType represents integer values only."
        pass

    @abstractmethod
    def is_fixed_point(self):
        "Returns whether this DataType represent fixed-point values only."
        pass

    @abstractmethod
    def get_hls_datatype_str(self):
        "Returns the corresponding Vivado HLS datatype name."
        pass

    @abstractmethod
    def to_numpy_dt(self):
        "Return an appropriate numpy datatype that can represent this QONNX DataType."
        pass

    @abstractmethod
    def get_canonical_name(self):
        "Return a canonical string representation of this QONNX DataType."


class FloatType(BaseDataType):
    def bitwidth(self):
        return 32

    def min(self):
        return np.finfo(np.float32).min

    def max(self):
        return np.finfo(np.float32).max

    def allowed(self, value):
        return True

    def get_num_possible_values(self):
        raise Exception("Undefined for FloatType")

    def is_integer(self):
        return False

    def is_fixed_point(self):
        return False

    def get_hls_datatype_str(self):
        return "float"

    def to_numpy_dt(self):
        return np.float32

    def get_canonical_name(self):
        return "FLOAT32"


class Float16Type(BaseDataType):
    def bitwidth(self):
        return 16

    def min(self):
        return np.finfo(np.float16).min

    def max(self):
        return np.finfo(np.float16).max

    def allowed(self, value):
        return True

    def get_num_possible_values(self):
        raise Exception("Undefined for Float16Type")

    def is_integer(self):
        return False

    def is_fixed_point(self):
        return False

    def get_hls_datatype_str(self):
        return "float"

    def to_numpy_dt(self):
        return np.float16

    def get_canonical_name(self):
        return "FLOAT16"


class IntType(BaseDataType):
    def __init__(self, bitwidth, signed):
        super().__init__()
        self._bitwidth = bitwidth
        self._signed = signed

    def bitwidth(self):
        return self._bitwidth

    def min(self):
        unsigned_min = 0
        signed_min = -(2 ** (self.bitwidth() - 1))
        return signed_min if self._signed else unsigned_min

    def max(self):
        unsigned_max = (2 ** (self.bitwidth())) - 1
        signed_max = (2 ** (self.bitwidth() - 1)) - 1
        return signed_max if self._signed else unsigned_max

    def allowed(self, value):
        return (self.min() <= value) and (value <= self.max()) and float(value).is_integer()

    def get_num_possible_values(self):
        return abs(self.min()) + abs(self.max()) + 1

    def is_integer(self):
        return True

    def is_fixed_point(self):
        return False

    def get_hls_datatype_str(self):
        if self.signed():
            return "ap_int<%d>" % self.bitwidth()
        else:
            return "ap_uint<%d>" % self.bitwidth()

    def to_numpy_dt(self):
        if self.bitwidth() <= 8:
            return np.int8 if self.signed() else np.uint8
        elif self.bitwidth() <= 16:
            return np.int16 if self.signed() else np.uint16
        elif self.bitwidth() <= 32:
            return np.int32 if self.signed() else np.uint32
        elif self.bitwidth() <= 64:
            return np.int64 if self.signed() else np.uint64
        else:
            raise Exception("Unknown numpy dtype for " + str(self))

    def get_canonical_name(self):
        if self.bitwidth() == 1 and (not self.signed()):
            return "BINARY"
        else:
            prefix = "INT" if self.signed() else "UINT"
            return prefix + str(self.bitwidth())


class BipolarType(BaseDataType):
    def bitwidth(self):
        return 1

    def min(self):
        return -1

    def max(self):
        return +1

    def allowed(self, value):
        return value in [-1, +1]

    def get_num_possible_values(self):
        return 2

    def is_integer(self):
        return True

    def is_fixed_point(self):
        return False

    def get_hls_datatype_str(self):
        return "ap_int<1>"

    def to_numpy_dt(self):
        return np.int8

    def get_canonical_name(self):
        return "BIPOLAR"


class TernaryType(BaseDataType):
    def bitwidth(self):
        return 2

    def min(self):
        return -1

    def max(self):
        return +1

    def allowed(self, value):
        return value in [-1, 0, +1]

    def get_num_possible_values(self):
        return 3

    def is_integer(self):
        return True

    def is_fixed_point(self):
        return False

    def get_hls_datatype_str(self):
        return "ap_int<2>"

    def to_numpy_dt(self):
        return np.int8

    def get_canonical_name(self):
        return "TERNARY"


class FixedPointType(IntType):
    def __init__(self, bitwidth, intwidth):
        super().__init__(bitwidth=bitwidth, signed=True)
        assert intwidth < bitwidth, "FixedPointType violates intwidth < bitwidth"
        self._intwidth = intwidth

    def int_bits(self):
        return self._intwidth

    def frac_bits(self):
        return self.bitwidth() - self.int_bits()

    def scale_factor(self):
        return 2 ** -(self.frac_bits())

    def min(self):
        return super().min() * self.scale_factor()

    def max(self):
        return super().max() * self.scale_factor()

    def allowed(self, value):
        int_value = value / self.scale_factor()
        return IntType(self._bitwidth, True).allowed(int_value)

    def is_integer(self):
        return False

    def is_fixed_point(self):
        return True

    def get_hls_datatype_str(self):
        return "ap_fixed<%d, %d>" % (self.bitwidth(), self.int_bits())

    def to_numpy_dt(self):
        return np.float32

    def get_canonical_name(self):
        return "FIXED<%d,%d>" % (self.bitwidth(), self.int_bits())


class ScaledIntType(IntType):
    # scaled integer datatype, only intended for
    # inference cost calculations, many of the
    # member methods are not implemented
    def __init__(self, bitwidth):
        super().__init__(bitwidth=bitwidth, signed=True)

    def min(self):
        raise Exception("Undefined for ScaledIntType")

    def max(self):
        raise Exception("Undefined for ScaledIntType")

    def allowed(self, value):
        raise Exception("Undefined for ScaledIntType")

    def is_integer(self):
        return False

    def is_fixed_point(self):
        return False

    def get_hls_datatype_str(self):
        raise Exception("Undefined for ScaledIntType")

    def to_numpy_dt(self):
        return np.float32

    def signed(self):
        "Returns whether this DataType can represent negative numbers."
        return True

    def get_canonical_name(self):
        return "SCALEDINT<%d>" % (self.bitwidth())


def resolve_datatype(name):
    _special_types = {
        "BINARY": IntType(1, False),
        "BIPOLAR": BipolarType(),
        "TERNARY": TernaryType(),
        "FLOAT32": FloatType(),
        "FLOAT16": Float16Type(),
    }
    if name in _special_types.keys():
        return _special_types[name]
    elif name.startswith("UINT"):
        bitwidth = int(name.replace("UINT", ""))
        return IntType(bitwidth, False)
    elif name.startswith("INT"):
        bitwidth = int(name.replace("INT", ""))
        return IntType(bitwidth, True)
    elif name.startswith("FIXED"):
        name = name.replace("FIXED<", "")
        name = name.replace(">", "")
        nums = name.split(",")
        bitwidth = int(nums[0].strip())
        intwidth = int(nums[1].strip())
        return FixedPointType(bitwidth, intwidth)
    elif name.startswith("SCALEDINT"):
        name = name.replace("SCALEDINT<", "")
        name = name.replace(">", "")
        nums = name.split(",")
        bitwidth = int(nums[0].strip())
        return ScaledIntType(bitwidth)
    else:
        raise KeyError("Could not resolve DataType " + name)


class DataTypeMeta(EnumMeta):
    def __getitem__(self, name):
        return resolve_datatype(name)


class DataType(Enum, metaclass=DataTypeMeta):
    """Enum class that contains QONNX data types to set the quantization annotation.
    ONNX does not support data types smaller than 8-bit integers, whereas in QONNX we are
    interested in smaller integers down to ternary and bipolar."""

    @staticmethod
    def get_accumulator_dt_cands():
        cands = ["BINARY"]
        cands += ["UINT%d" % (x + 1) for x in range(64)]
        cands += ["BIPOLAR", "TERNARY"]
        cands += ["INT%d" % (x + 1) for x in range(64)]
        return cands

    @staticmethod
    def get_smallest_possible(value):
        """Returns smallest (fewest bits) possible DataType that can represent
        value. Prefers unsigned integers where possible."""
        if not int(value) == value:
            return DataType["FLOAT32"]
        cands = DataType.get_accumulator_dt_cands()
        for cand in cands:
            dt = DataType[cand]
            if (dt.min() <= value) and (value <= dt.max()):
                return dt
        raise Exception("Could not find a suitable int datatype for " + str(value))
