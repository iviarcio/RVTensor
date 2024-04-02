//===- RVMathDialect.cpp - RVMath dialect ---------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "RVMath/RVMathDialect.h"
#include "RVMath/RVMathOps.h"
#include "RVMath/RVMathTypes.h"

using namespace mlir;
using namespace mlir::rvmath;

#include "RVMath/RVMathOpsDialect.cpp.inc"

//===----------------------------------------------------------------------===//
// RVMath dialect initialization.
//===----------------------------------------------------------------------===//

void RVMathDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "RVMath/RVMathOps.cpp.inc"
      >();
  registerTypes();
}
