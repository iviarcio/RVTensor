//===- RVTensorTypes.cpp - RVTensor dialect types -----------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "RVTensor/RVTensorTypes.h"

#include "RVTensor/RVTensorDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir::rvtensor;

#define GET_TYPEDEF_CLASSES
#include "RVTensor/RVTensorOpsTypes.cpp.inc"

void RVTensorDialect::registerTypes() {
  addTypes<
#define GET_TYPEDEF_LIST
#include "RVTensor/RVTensorOpsTypes.cpp.inc"
      >();
}
