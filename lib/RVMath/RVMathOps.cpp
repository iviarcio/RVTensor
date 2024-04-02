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

using namespace mlir;
using namespace mlir::rvmath;

//===----------------------------------------------------------------------===//
// RVMath Operator Verifiers.
//===----------------------------------------------------------------------===//

static bool hasZeroDimension(ShapedType shapedType) {
  if (!shapedType.hasRank())
    return false;

  auto rank = shapedType.getRank();

  for (int i = 0; i < rank; i++) {
    if (shapedType.isDynamicDim(i))
      continue;
    if (shapedType.getDimSize(i) == 0)
      return true;
  }

  return false;
}

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

} // verifyConvOp

LogicalResult Conv2DOp::verify() { return verifyConvOp(*this); }

LogicalResult Conv3DOp::verify() { return verifyConvOp(*this); }

LogicalResult DepthwiseConv2DOp::verify() { return verifyConvOp(*this); }

LogicalResult FullyConnectedOp::verify() { return verifyConvOp(*this); }

//===----------------------------------------------------------------------===//
// RVMath Operator Quantization Builders.
//===----------------------------------------------------------------------===//

/// This builder is called on all convolution operators except TransposeConv,
/// which has specialized output shape semantics. The builder also defines the
/// bitwidth of the output given the bit width of the input & weight content.
static void buildConvOpWithQuantInfo(OpBuilder &builder, OperationState &result,
                                     Type outputType, Value input, Value weight,
                                     Value bias, DenseI64ArrayAttr pad,
                                     DenseI64ArrayAttr stride,
                                     DenseI64ArrayAttr dilation) {

  result.addOperands({input, weight, bias});
  result.addAttribute("pad", pad);
  result.addAttribute("stride", stride);
  result.addAttribute("dilation", dilation);

  auto quantAttr = buildConvOpQuantizationAttr(builder, input, weight);
  if (quantAttr) {
    result.addAttribute("quantization_info", quantAttr);
    result.addTypes(
        buildConvOpResultTypeInfo(builder, outputType, input, weight));
  } else {
    result.addTypes(outputType);
  }
}

/// Handles rvmath.transpose_conv2d which has outpad and output shape attributes.
static void buildTransConvOpWithQuantInfo(
    OpBuilder &builder, OperationState &result, Type outputType, Value input,
    Value weight, Value bias, DenseI64ArrayAttr outpad,
    DenseI64ArrayAttr stride, DenseI64ArrayAttr outputShape) {
  result.addOperands({input, weight, bias});
  result.addAttribute("out_pad", outpad);
  result.addAttribute("stride", stride);
  result.addAttribute("out_shape", outputShape);
  auto quantAttr = ::buildConvOpQuantizationAttr(builder, input, weight);

  if (quantAttr) {
    result.addAttribute("quantization_info", quantAttr);
    result.addTypes(
        buildConvOpResultTypeInfo(builder, outputType, input, weight));
  } else {
    result.addTypes(outputType);
  }
}

/// The rvmath.fully_connected op has its own builder as it does not have
/// strides/dilation/padding.
static void buildFCOpWithQuantInfo(OpBuilder &builder, OperationState &result,
                                   Type outputType, Value input, Value weight,
                                   Value bias) {

  result.addOperands({input, weight, bias});
  auto quantAttr = ::buildConvOpQuantizationAttr(builder, input, weight);
  if (quantAttr) {
    result.addAttribute("quantization_info", quantAttr);
    result.addTypes(
        buildConvOpResultTypeInfo(builder, outputType, input, weight));
  } else {
    result.addTypes(outputType);
  }
}

/// The rvmath.matmul op is also intended to be generated where a fully_connected
/// op must be constructed where the weight is not a constant. In this case,
/// the fully_connected op must be expressed using matmul.
static void buildMatMulOpWithQuantInfo(OpBuilder &builder,
                                       OperationState &result, Type outputType,
                                       Value a, Value b) {
  result.addOperands({a, b});
  auto quantAttr = ::buildMatMulOpQuantizationAttr(builder, a, b);

  if (quantAttr) {
    result.addAttribute("quantization_info", quantAttr);

    auto inputType = llvm::dyn_cast<ShapedType>(a.getType());
    assert(inputType && "Input must be a shaped tensor type!");

    auto inputQType = llvm::dyn_cast<mlir::quant::UniformQuantizedType>(
        inputType.getElementType());
    assert(inputQType && "Tensor must have quantized datatype!");

    unsigned inputBits = inputQType.getStorageTypeIntegralWidth();

    auto outputShapedType = llvm::dyn_cast<ShapedType>(outputType);
    assert(outputShapedType && "Output must be a shaped type");

    IntegerType accElementType;
    if (inputBits == 16)
      accElementType = builder.getIntegerType(48);
    else
      accElementType = builder.getI32Type();
    auto accType = outputShapedType.clone(accElementType);
    result.addTypes(accType);
  } else {
    result.addTypes(outputType);
  }
}

//===----------------------------------------------------------------------===//
// RVMath Infer Return Type Components.
//===----------------------------------------------------------------------===//

LogicalResult Conv2DOp::inferReturnTypeComponents(
    MLIRContext *context, ::std::optional<Location> location,
    Conv2DOp::Adaptor adaptor,
    SmallVectorImpl<ShapedTypeComponents> &inferredReturnShapes) {
  llvm::SmallVector<int64_t> outputShape(4, ShapedType::kDynamic);

  int64_t inputWidth = ShapedType::kDynamic;
  int64_t inputHeight = ShapedType::kDynamic;
  int64_t weightWidth = ShapedType::kDynamic;
  int64_t weightHeight = ShapedType::kDynamic;

  // Input shape describes input width/height and batch.

  ShapeAdaptor inputShape(adaptor.getInput().getType());
  if (inputShape.hasRank()) {
    outputShape[0] = inputShape.getDimSize(0);
    inputHeight = inputShape.getDimSize(1);
    inputWidth = inputShape.getDimSize(2);
  }

  // Weight shapes describes the filter width/height and the output channels.
  ShapeAdaptor weightShape(adaptor.getWeight().getType());
  if (weightShape.hasRank()) {
    outputShape[3] = weightShape.getDimSize(0);
    weightHeight = weightShape.getDimSize(1);
    weightWidth = weightShape.getDimSize(2);
  }

  // Bias shape can describe the output channels.
  ShapeAdaptor biasShape(adaptor.getBias().getType());
  if (biasShape.hasRank()) {
    outputShape[3] = ShapedType::isDynamic(outputShape[3])
                         ? biasShape.getDimSize(0)
                         : outputShape[3];
  }

  llvm::ArrayRef<int64_t> dilation = adaptor.getDilation();
  llvm::ArrayRef<int64_t> stride = adaptor.getStride();
  llvm::ArrayRef<int64_t> padding = adaptor.getPad();

  if (!ShapedType::isDynamic(inputHeight) &&
      !ShapedType::isDynamic(weightHeight)) {
    int64_t inputSize = inputHeight + padding[0] + padding[1];
    int64_t filterSize = (weightHeight - 1) * dilation[0] + 1;
    int64_t unstridedResult = inputSize - filterSize + 1;
    outputShape[1] = (unstridedResult - 1) / stride[0] + 1;
  }

  if (!ShapedType::isDynamic(inputWidth) &&
      !ShapedType::isDynamic(weightWidth)) {
    int64_t inputSize = inputWidth + padding[2] + padding[3];
    int64_t filterSize = (weightWidth - 1) * dilation[1] + 1;
    int64_t unstridedResult = inputSize - filterSize + 1;
    outputShape[2] = (unstridedResult - 1) / stride[1] + 1;
  }

  inferredReturnShapes.push_back(ShapedTypeComponents(outputShape));
  return success();

} // Conv2DOp

LogicalResult DepthwiseConv2DOp::inferReturnTypeComponents(
    MLIRContext *context, ::std::optional<Location> location,
    DepthwiseConv2DOp::Adaptor adaptor,
    SmallVectorImpl<ShapedTypeComponents> &inferredReturnShapes) {
  llvm::SmallVector<int64_t> outputShape(4, ShapedType::kDynamic);

  int64_t inputWidth = ShapedType::kDynamic;
  int64_t inputHeight = ShapedType::kDynamic;
  int64_t inputChannels = ShapedType::kDynamic;

  int64_t weightWidth = ShapedType::kDynamic;
  int64_t weightHeight = ShapedType::kDynamic;
  int64_t depthChannels = ShapedType::kDynamic;

  // Input shape describes input width/height and batch.
  ShapeAdaptor inputShape(adaptor.getInput().getType());
  if (inputShape.hasRank()) {
    outputShape[0] = inputShape.getDimSize(0);
    inputHeight = inputShape.getDimSize(1);
    inputWidth = inputShape.getDimSize(2);
    inputChannels = inputShape.getDimSize(3);
  }

  // Weight shapes describes the filter width/height and the output channels.
  ShapeAdaptor weightShape(adaptor.getWeight().getType());
  if (weightShape.hasRank()) {
    weightHeight = weightShape.getDimSize(0);
    weightWidth = weightShape.getDimSize(1);
    inputChannels = ShapedType::isDynamic(inputChannels)
                        ? weightShape.getDimSize(2)
                        : inputChannels;
    depthChannels = weightShape.getDimSize(3);
  }

  // If both inputChannels and depthChannels are available we can determine
  // the output channels.
  if (!ShapedType::isDynamic(inputChannels) &&
      !ShapedType::isDynamic(depthChannels)) {
    outputShape[3] = inputChannels * depthChannels;
  }

  // Bias shape can describe the output channels.
  ShapeAdaptor biasShape(adaptor.getBias().getType());
  if (biasShape.hasRank()) {
    outputShape[3] = ShapedType::isDynamic(outputShape[3])
                         ? biasShape.getDimSize(0)
                         : outputShape[3];
  }

  llvm::ArrayRef<int64_t> dilation = adaptor.getDilation();
  llvm::ArrayRef<int64_t> padding = adaptor.getPad();
  llvm::ArrayRef<int64_t> stride = adaptor.getStride();

  if (!ShapedType::isDynamic(inputHeight) &&
      !ShapedType::isDynamic(weightHeight)) {
    int64_t inputSize = inputHeight + padding[0] + padding[1];
    int64_t filterSize = (weightHeight - 1) * dilation[0] + 1;
    int64_t unstridedResult = inputSize - filterSize + 1;
    outputShape[1] = (unstridedResult - 1) / stride[0] + 1;
  }

  if (!ShapedType::isDynamic(inputWidth) &&
      !ShapedType::isDynamic(weightWidth)) {
    int64_t inputSize = inputWidth + padding[2] + padding[3];
    int64_t filterSize = (weightWidth - 1) * dilation[1] + 1;
    int64_t unstridedResult = inputSize - filterSize + 1;
    outputShape[2] = (unstridedResult - 1) / stride[1] + 1;
  }

  inferredReturnShapes.push_back(ShapedTypeComponents(outputShape));
  return success();

} //DepthwiseConv2DOp

LogicalResult rvmath::FullyConnectedOp::inferReturnTypeComponents(
    MLIRContext *context, ::std::optional<Location> location,
    FullyConnectedOp::Adaptor adaptor,
    SmallVectorImpl<ShapedTypeComponents> &inferredReturnShapes) {
  ShapeAdaptor inputShape(adaptor.getInput().getType());
  ShapeAdaptor weightShape(adaptor.getWeight().getType());
  ShapeAdaptor biasShape(adaptor.getBias().getType());

  // All shapes are dynamic.
  SmallVector<int64_t> outShape;
  outShape.resize(2, ShapedType::kDynamic);

  if (inputShape.hasRank()) {
    outShape[0] = inputShape.getDimSize(0);
  }

  if (weightShape.hasRank()) {
    outShape[1] = weightShape.getDimSize(0);
  }

  if (biasShape.hasRank()) {
    outShape[1] = outShape[1] == ShapedType::kDynamic ? biasShape.getDimSize(0)
                                                      : outShape[1];
  }

  inferredReturnShapes.push_back(ShapedTypeComponents(outShape));
  return success();

} // FullyConnectedOp

LogicalResult rvmath::MatMulOp::inferReturnTypeComponents(
    MLIRContext *context, ::std::optional<Location> location,
    MatMulOp::Adaptor adaptor,
    SmallVectorImpl<ShapedTypeComponents> &inferredReturnShapes) {
  ShapeAdaptor lhsShape(adaptor.getA().getType());
  ShapeAdaptor rhsShape(adaptor.getB().getType());

  // All shapes are dynamic.
  SmallVector<int64_t> outShape;
  outShape.resize(3, ShapedType::kDynamic);

  if (lhsShape.hasRank()) {
    outShape[0] = lhsShape.getDimSize(0);
    outShape[1] = lhsShape.getDimSize(1);
  }

  if (rhsShape.hasRank()) {
    outShape[0] = outShape[0] == ShapedType::kDynamic ? rhsShape.getDimSize(0)
                                                      : outShape[0];
    outShape[2] = rhsShape.getDimSize(2);
  }

  inferredReturnShapes.push_back(ShapedTypeComponents(outShape));
  return success();

} // MatMulOp

LogicalResult Conv3DOp::inferReturnTypeComponents(
    MLIRContext *context, ::std::optional<Location> location,
    Conv3DOp::Adaptor adaptor,
    SmallVectorImpl<ShapedTypeComponents> &inferredReturnShapes) {
  llvm::SmallVector<int64_t> outputShape(5, ShapedType::kDynamic);

  int64_t inputWidth = ShapedType::kDynamic;
  int64_t inputHeight = ShapedType::kDynamic;
  int64_t inputDepth = ShapedType::kDynamic;

  int64_t weightWidth = ShapedType::kDynamic;
  int64_t weightHeight = ShapedType::kDynamic;
  int64_t weightDepth = ShapedType::kDynamic;

  // Input shape describes input width/height and batch.
  ShapeAdaptor inputShape(adaptor.getInput().getType());
  if (inputShape.hasRank()) {
    outputShape[0] = inputShape.getDimSize(0);
    inputDepth = inputShape.getDimSize(1);
    inputHeight = inputShape.getDimSize(2);
    inputWidth = inputShape.getDimSize(3);
  }

  // Weight shapes describes the filter width/height and the output channels.
  ShapeAdaptor weightShape(adaptor.getWeight().getType());
  if (weightShape.hasRank()) {
    outputShape[4] = weightShape.getDimSize(0);
    weightDepth = weightShape.getDimSize(1);
    weightHeight = weightShape.getDimSize(2);
    weightWidth = weightShape.getDimSize(3);
  }

  // Bias shape can describe the output channels.
  ShapeAdaptor biasShape(adaptor.getBias().getType());
  if (biasShape.hasRank() && ShapedType::isDynamic(outputShape[4])) {
    outputShape[4] = biasShape.getDimSize(0);
  }

  llvm::ArrayRef<int64_t> dilation = adaptor.getDilation();
  llvm::ArrayRef<int64_t> stride = adaptor.getStride();
  llvm::ArrayRef<int64_t> pad = adaptor.getPad();

  if (!ShapedType::isDynamic(inputDepth) &&
      !ShapedType::isDynamic(weightDepth)) {
    int32_t inputSize = inputDepth + pad[0] + pad[1];
    int32_t filterSize = (weightDepth - 1) * dilation[0] + 1;
    int32_t unstridedResult = inputSize - filterSize + 1;
    outputShape[1] = (unstridedResult - 1) / stride[0] + 1;
  }

  if (!ShapedType::isDynamic(inputHeight) &&
      !ShapedType::isDynamic(weightHeight)) {
    int32_t inputSize = inputHeight + pad[2] + pad[3];
    int32_t filterSize = (weightHeight - 1) * dilation[1] + 1;
    int32_t unstridedResult = inputSize - filterSize + 1;
    outputShape[2] = (unstridedResult - 1) / stride[1] + 1;
  }

  if (!ShapedType::isDynamic(inputWidth) &&
      !ShapedType::isDynamic(weightWidth)) {
    int32_t inputSize = inputWidth + pad[4] + pad[5];
    int32_t filterSize = (weightWidth - 1) * dilation[2] + 1;
    int32_t unstridedResult = inputSize - filterSize + 1;
    outputShape[3] = (unstridedResult - 1) / stride[2] + 1;
  }

  inferredReturnShapes.push_back(ShapedTypeComponents(outputShape));
  return success();

} // Conv3DOp

LogicalResult TransposeConv2DOp::inferReturnTypeComponents(
    MLIRContext *context, ::std::optional<Location> location,
    TransposeConv2DOp::Adaptor adaptor,
    SmallVectorImpl<ShapedTypeComponents> &inferredReturnShapes) {
  // outputShape is mutable.
  llvm::SmallVector<int64_t> outputShape =
      convertToMlirShape(adaptor.getOutShape());

  int64_t inputWidth = ShapedType::kDynamic;
  int64_t inputHeight = ShapedType::kDynamic;
  int64_t weightWidth = ShapedType::kDynamic;
  int64_t weightHeight = ShapedType::kDynamic;

  // Input shape describes input width/height and batch.
  ShapeAdaptor inputShape(adaptor.getInput().getType());
  if (inputShape.hasRank()) {
    outputShape[0] = ShapedType::isDynamic(outputShape[0])
                         ? inputShape.getDimSize(0)
                         : outputShape[0];
    inputHeight = inputShape.getDimSize(1);
    inputWidth = inputShape.getDimSize(2);
  }

  // Weight shapes describes the filter width/height and the output channels.
  ShapeAdaptor weightShape(adaptor.getFilter().getType());
  if (weightShape.hasRank()) {
    outputShape[3] = ShapedType::isDynamic(outputShape[3])
                         ? weightShape.getDimSize(0)
                         : outputShape[3];
    weightHeight = weightShape.getDimSize(1);
    weightWidth = weightShape.getDimSize(2);
  }

  // Bias shape can describe the output channels.
  ShapeAdaptor biasShape(adaptor.getInput().getType());
  if (biasShape.hasRank()) {
    outputShape[3] = ShapedType::isDynamic(outputShape[3])
                         ? biasShape.getDimSize(0)
                         : outputShape[3];
  }

  llvm::ArrayRef<int64_t> padding = adaptor.getOutPad();
  llvm::ArrayRef<int64_t> stride = adaptor.getStride();

  if (!ShapedType::isDynamic(inputHeight) &&
      !ShapedType::isDynamic(weightHeight)) {
    int64_t calculateSize =
        (inputHeight - 1) * stride[0] + padding[0] + padding[1] + weightHeight;
    outputShape[1] =
        ShapedType::isDynamic(outputShape[1]) ? calculateSize : outputShape[1];
  }

  if (!ShapedType::isDynamic(inputWidth) &&
      !ShapedType::isDynamic(weightWidth)) {
    int64_t calculateSize =
        (inputWidth - 1) * stride[1] + padding[2] + padding[3] + weightWidth;
    outputShape[2] =
        ShapedType::isDynamic(outputShape[2]) ? calculateSize : outputShape[2];
  }

  inferredReturnShapes.push_back(ShapedTypeComponents(outputShape));
  return success();

} // TransposeConv2DOp
