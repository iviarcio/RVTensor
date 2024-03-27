//===- RVMathTypes.cpp - RVMath dialect types -----------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "RVMath/RVMathTypes.h"

#include "RVMath/RVMathDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir::rvmath;

#define GET_TYPEDEF_CLASSES
#include "RVMath/RVMathOpsTypes.cpp.inc"

void RVMathDialect::registerTypes() {
  addTypes<
#define GET_TYPEDEF_LIST
#include "RVMath/RVMathOpsTypes.cpp.inc"
      >();
}
