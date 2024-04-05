//===- RVTensorDialect.cpp - RVTensor dialect ---------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "RVTensor/RVTensorDialect.h"
#include "RVTensor/RVTensorOps.h"
#include "RVTensor/RVTensorTypes.h"

using namespace mlir;
using namespace mlir::rvtensor;

#include "RVTensor/RVTensorDialect.cpp.inc"

//===----------------------------------------------------------------------===//
// RVTensor dialect initialization.
//===----------------------------------------------------------------------===//

void RVTensorDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "RVTensor/RVTensorOps.cpp.inc"
      >();
  registerTypes();
}
