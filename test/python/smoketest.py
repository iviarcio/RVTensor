# RUN: %python %s | FileCheck %s

from mlir_rvtensor.ir import *
from mlir_rvtensor.dialects import builtin as builtin_d, rvtensor as rvtensor_d

with Context():
    rvtensor_d.register_dialect()
    module = Module.parse(
        """
    %0 = arith.constant 2 : i32
    %1 = rvtensor.foo %0 : i32
    """
    )
    # CHECK: %[[C:.*]] = arith.constant 2 : i32
    # CHECK: rvtensor.foo %[[C]] : i32
    print(str(module))
