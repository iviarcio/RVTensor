//===- RVMathPasses.h - RVMath passes  ------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#ifndef RVMATH_RVMATHPASSES_H
#define RVMATH_RVMATHPASSES_H

#include "RVMath/RVMathDialect.h"
#include "RVMath/RVMathOps.h"
#include "mlir/Pass/Pass.h"
#include <memory>

namespace mlir {
namespace rvmath {
#define GEN_PASS_DECL
#include "RVMath/RVMathPasses.h.inc"

#define GEN_PASS_REGISTRATION
#include "RVMath/RVMathPasses.h.inc"
} // namespace rvmath
} // namespace mlir

#endif
