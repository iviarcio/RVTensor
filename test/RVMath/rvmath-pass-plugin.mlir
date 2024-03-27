// RUN: mlir-opt %s --load-pass-plugin=%rvmath_libs/RVMathPlugin%shlibext --pass-pipeline="builtin.module(rvmath-switch-bar-foo)" | FileCheck %s

module {
  // CHECK-LABEL: func @foo()
  func.func @bar() {
    return
  }

  // CHECK-LABEL: func @abar()
  func.func @abar() {
    return
  }
}
