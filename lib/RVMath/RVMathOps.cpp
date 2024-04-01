//===- RVMathOps.cpp - RVMath dialect ops ---------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "RVMath/RVMathOps.h"
#include "RVMath/RVMathDialect.h"
#include "mlir/IR/OpBase.td"
#include "mlir/Support/LogicalResult.h"

#define GET_OP_CLASSES
#include "RVMath/RVMathOps.cpp.inc"

namespace mlir {

template <typename T> static LogicalResult verifyConvOp(T op) {
  // All RVMath conv ops have an input() and weight().
  auto inputType = llvm::dyn_cast<RankedTensorType>(op.getInput().getType());
  auto weightType = llvm::dyn_cast<RankedTensorType>(op.getWeight().getType());

  // Must be ranked tensor types
  if (!inputType) {
    op.emitOpError("expect a ranked tensor for input, got ") << op.getInput();
    return failure();
  }
  if (!weightType) {
    op.emitOpError("expect a ranked tensor for weight, got ") << op.getWeight();
    return failure();
  }

  if (hasZeroDimension(inputType))
    return op.emitOpError() << "tensor has a dimension with size zero. Each "
                               "dimension of a tensor must have size >= 1";

  auto inputEType = inputType.getElementType();
  auto weightEType = weightType.getElementType();

  bool inputIsQuant = !llvm::isa<FloatType>(inputEType);
  bool weightIsQuant = !llvm::isa<FloatType>(weightEType);

  // Either both must be quantized or both unquantized.
  if (inputIsQuant != weightIsQuant) {
    op.emitOpError(
        "expect both input and weight to be float or not together, got ")
        << inputEType << " and " << weightEType;
    return failure();
  }

  // Quantized type must have constructed the quantizationattr, and unquantized
  // types should not have a quantizationattr.
  if ((inputIsQuant && !op.getQuantizationInfo()) ||
      (!inputIsQuant && op.getQuantizationInfo())) {
    op.emitOpError("quantizationattr is required for quantized type, and not "
                   "allowed for float type");
    return failure();
  }

  return success();
}

LogicalResult Conv2DOp::verify() { return verifyConvOp(*this); }

LogicalResult Conv3DOp::verify() { return verifyConvOp(*this); }

LogicalResult DepthwiseConv2DOp::verify() { return verifyConvOp(*this); }

LogicalResult FullyConnectedOp::verify() { return verifyConvOp(*this); }

}
