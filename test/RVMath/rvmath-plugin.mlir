// RUN: mlir-opt %s --load-dialect-plugin=%rvmath_libs/RVMathPlugin%shlibext --pass-pipeline="builtin.module(rvmath-switch-bar-foo)" | FileCheck %s

module {
  // CHECK-LABEL: func @foo()
  func.func @bar() {
    return
  }

  // CHECK-LABEL: func @rvmath_types(%arg0: !rvmath.custom<"10">)
  func.func @rvmath_types(%arg0: !rvmath.custom<"10">) {
    return
  }
}
