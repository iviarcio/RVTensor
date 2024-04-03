//===- RVTensorPasses.h - RVTensor passes  ------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#ifndef RVTENSOR_RVTENSORPASSES_H
#define RVTENSOR_RVTENSORPASSES_H

#include "RVTensor/RVTensorDialect.h"
#include "RVTensor/RVTensorOps.h"
#include "mlir/Pass/Pass.h"
#include <memory>

namespace mlir {
namespace rvtensor {
#define GEN_PASS_DECL
#include "RVTensor/RVTensorPasses.h.inc"

#define GEN_PASS_REGISTRATION
#include "RVTensor/RVTensorPasses.h.inc"
} // namespace rvtensor
} // namespace mlir

#endif
