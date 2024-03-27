# RUN: %python %s | FileCheck %s

from mlir_rvmath.ir import *
from mlir_rvmath.dialects import builtin as builtin_d, rvmath as rvmath_d

with Context():
    rvmath_d.register_dialect()
    module = Module.parse(
        """
    %0 = arith.constant 2 : i32
    %1 = rvmath.foo %0 : i32
    """
    )
    # CHECK: %[[C:.*]] = arith.constant 2 : i32
    # CHECK: rvmath.foo %[[C]] : i32
    print(str(module))
