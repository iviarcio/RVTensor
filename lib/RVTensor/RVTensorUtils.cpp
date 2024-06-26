//===- QuantUtils.cpp -----------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains RVTENSOR numerical support functions and quantization
// attribute builders.
//
//===----------------------------------------------------------------------===//

#include "RVTensor/RVTensorUtils.h"

using namespace mlir;
using namespace mlir::rvtensor;

/// From a scale value, generates multiplier and shift values where
/// mantissa is in [-1.0,-0.5] or [0.5, 1.0] such that
/// multiplier = mantissa*2^shift for 16-bit scaling.
static void computeMultiplierAndShiftRVTensorScale16(double scale,
                                                 int32_t &multiplier,
                                                 int32_t &shift) {

  const double mantissa = std::frexp(scale, &shift);
  auto shiftedM = std::round(mantissa * (int64_t(1) << 15));

  // Can't be greater than 1.0.
  assert(shiftedM <= (int64_t(1) << 15) &&
         "Shifted mantissa exceeds 16 signed bits");

  if (shiftedM == (int64_t(1) << 15)) {
    shiftedM /= 2;
    shift++;
  }

  // RVTENSOR expects right shift to be positive and embed (1 << 15) into right
  // shift bits.
  shift = (-shift) + 15;

  assert(shiftedM <= std::numeric_limits<int32_t>::max() &&
         "Shifted mantissa exceeds 32-bit signed output type");

  multiplier = static_cast<int32_t>(shiftedM);

  // Shifting tops out at 62 bits. Right shift to make 62 bits the max.
  // The limit of 62 on shift allows the shift to be decomposed as
  // two right shifts of 31.
  if (shift > 62) {
    // Shifting the multiplier by more than 31-bits is unnecessary.
    multiplier = multiplier >> std::min<int32_t>(31, shift - 62);
    shift = 62;
  }
}

/// From a scale value, generates multiplier and shift values where
/// mantissa is in [-1.0,-0.5] or [0.5, 1.0] such that
/// multiplier = mantissa*2^shift for 32-bit scaling.
static void computeMultiplierAndShiftRVTensorScale32(double scale,
                                                 int32_t &multiplier,
                                                 int32_t &shift) {

  const double mantissa = std::frexp(scale, &shift);
  auto shiftedM = std::round(mantissa * (int64_t(1) << 31));

  // Can't be greater than 1.0.
  assert(shiftedM <= (int64_t(1) << 31) &&
         "Shifted mantissa exceeds 32 signed bits");
  if (shiftedM == (int64_t(1) << 31)) {
    shiftedM /= 2;
    shift++;
  }

  // RVTENSOR expects right shift to be positive, and embed (1 << 31) into right
  // shift bits.
  shift = (-shift) + 31;

  assert(shiftedM <= std::numeric_limits<int32_t>::max() &&
         "Shifted mantissa exceeds 32-bit signed output type");

  multiplier = static_cast<int32_t>(shiftedM);

  // Shifting tops out at 62 bits. Right shift to make 62 bits the max.
  // The limit of 62 on shift allows the shift to be decomposed as
  // two right shifts of 31.
  if (shift > 62) {
    // Shifting the multiplier by more than 32-bits is unnecessary.
    multiplier = multiplier >> std::min<int32_t>(31, shift - 62);
    shift = 62;
  }
}

/// Generates a quantized multiplier/shift from double.
void mlir::rvtensor::computeMultiplierAndShift(double scale, int32_t &multiplier,
                                           int32_t &shift, int32_t scaleWidth) {

  switch (scaleWidth) {
  case 16:
    computeMultiplierAndShiftRVTensorScale16(scale, multiplier, shift);
    return;
  case 32:
    computeMultiplierAndShiftRVTensorScale32(scale, multiplier, shift);
    return;
  default:
    assert(0 && "Unsupported RVTensor quantized_scale regime specified!");
  }
}

#define GET_UQTYPE(inputType)                                                  \
  (llvm::dyn_cast<quant::UniformQuantizedType>((inputType).getElementType()))
#define GET_QTYPE(inputType)                                                   \
  (llvm::dyn_cast<quant::QuantizedType>((inputType).getElementType()))

/// Method to build ConvOpQuantizationAttr, called from
/// ConvOpQuantInfoBuilder/TransConvOpQuantInfoBuilder:
/// input_zp: input zeropoint
/// weight_zp: weight zeropoint.
ConvOpQuantizationAttr
mlir::rvtensor::buildConvOpQuantizationAttr(OpBuilder &builder, Value input,
                                        Value weight) {

  auto inputType = dyn_cast<ShapedType>(input.getType());
  auto weightType = dyn_cast<ShapedType>(weight.getType());

  if (!inputType || !weightType)
    return nullptr;

  auto inputQType = GET_UQTYPE(inputType);
  auto weightPerTensorQType = GET_UQTYPE(weightType);
  auto weightPerAxisQType =
      dyn_cast<quant::UniformQuantizedPerAxisType>(weightType.getElementType());

  // Weights must be either per-tensor quantized or per-axis quantized.
  assert(!((bool)weightPerTensorQType && (bool)weightPerAxisQType) &&
         "Weights must be either per-tensor or per-axis quantized");

  // Either all quantized or all not quantized.
  assert(!((bool)inputQType ^
           ((bool)weightPerTensorQType || (bool)weightPerAxisQType)) &&
         "Inputs and weights must be all quantized or all not quantized");

  if (inputQType) {
    int64_t inputZp = inputQType.getZeroPoint();
    int64_t weightZp = 0;

    if (weightPerTensorQType) {
      weightZp = weightPerTensorQType.getZeroPoint();
    } else if (weightPerAxisQType) {
      weightZp = weightPerAxisQType.getZeroPoints().front();
    }

    return builder.getAttr<rvtensor::ConvOpQuantizationAttr>(inputZp, weightZp);
  }

  return nullptr;
}

/// Builds MatMulOpQuantizationAttr, called from
/// MatMulOpQuantInfoBuilder:
/// aZp: input a zeropoint
/// bZp: input b zeropoint.
MatMulOpQuantizationAttr
mlir::rvtensor::buildMatMulOpQuantizationAttr(OpBuilder &builder, Value a,
                                          Value b) {

  auto aType = dyn_cast<ShapedType>(a.getType());
  auto bType = dyn_cast<ShapedType>(b.getType());

  if (!aType || !bType)
    return nullptr;

  auto aQType = GET_UQTYPE(aType);
  auto bQType = GET_UQTYPE(bType);

  // A and B are either all quantized or all not quantized.
  assert(!((bool)aQType ^ (bool)bQType) &&
         "Matmul operands must be all quantized or all not quantized");

  if (aQType) {
    return builder.getAttr<rvtensor::MatMulOpQuantizationAttr>(
        aQType.getZeroPoint(), bQType.getZeroPoint());
  }

  return nullptr;
}

/// Builds UnaryOpQuantizationAttr
/// UnaryOpQuantInfoBuilder:
/// inputZp: input zeropoint
/// outputZp: output zeropoint.
UnaryOpQuantizationAttr
mlir::rvtensor::buildUnaryOpQuantizationAttr(OpBuilder &builder, Value input,
                                         Type outputRawType) {

  auto inputType = dyn_cast<ShapedType>(input.getType());
  auto outputType = dyn_cast<ShapedType>(outputRawType);

  if (!inputType || !outputType)
    return nullptr;

  auto inputQType = GET_UQTYPE(inputType);
  auto outputQType = GET_UQTYPE(outputType);

  // Either all quantized or all not quantized.
  assert(!((bool)inputQType ^ (bool)outputQType) &&
         "Unary inputs/outputs must be all quantized or all not quantized");

  if (inputQType) {
    return builder.getAttr<UnaryOpQuantizationAttr>(inputQType.getZeroPoint(),
                                                    outputQType.getZeroPoint());
  }

  return nullptr;
}

/// Builds PadOpQuantizationAttr, called from PadOpQuantInfoBuilder:
/// inputZp: input zeropoint.
PadOpQuantizationAttr mlir::rvtensor::buildPadOpQuantizationAttr(OpBuilder &builder,
                                                             Value input) {

  auto inputType = dyn_cast<ShapedType>(input.getType());

  if (!inputType)
    return nullptr;

  auto inputQType = GET_UQTYPE(inputType);

  if (inputQType) {
    return builder.getAttr<rvtensor::PadOpQuantizationAttr>(
        inputQType.getZeroPoint());
  }

  return nullptr;
}

/// Builds output type for a quantized ConvOp with the right bitwidth.
/// This is called by the builder when dealing with quantized content.
Type mlir::rvtensor::buildConvOpResultTypeInfo(OpBuilder &builder, Type outputType,
                                           Value input, Value weight) {

  auto inputType = dyn_cast<ShapedType>(input.getType());
  auto weightType = dyn_cast<ShapedType>(weight.getType());

  assert(inputType && weightType &&
         "Could not extract input or weight tensors from Conv op");

  auto inputQType = GET_QTYPE(inputType);
  auto weightQType = GET_QTYPE(weightType);

  assert(inputQType && weightQType &&
         "Could not extract input or weight tensor types from Conv op");

  unsigned inputBits = inputQType.getStorageTypeIntegralWidth();
  unsigned weightBits = weightQType.getStorageTypeIntegralWidth();

  auto outputShapedType = dyn_cast<ShapedType>(outputType);
  assert(outputShapedType &&
         "Could not extract output shape type from Conv op");

  IntegerType accElementType;
  if (inputBits == 16 && weightBits == 8)
    accElementType = builder.getIntegerType(48);
  else
    accElementType = builder.getI32Type();
  auto accType = outputShapedType.clone(accElementType);
  return accType;
}

/// Builds RVTensor quantization attributes from min/max values.
Type mlir::rvtensor::buildQTypeFromMinMax(OpBuilder builder, Type inputDType,
                                      Attribute minAttr, Attribute maxAttr,
                                      IntegerAttr quantBits, int filterQuantDim,
                                      bool isSigned, BoolAttr narrowRange) {

  quant::QuantizedType retType;

  auto convfunc =
      quant::ExpressedToQuantizedConverter::forInputType(inputDType);

  auto minElems = dyn_cast<DenseFPElementsAttr>(minAttr);
  auto maxElems = dyn_cast<DenseFPElementsAttr>(maxAttr);

  SmallVector<double, 2> min, max;

  // At least one is per-axis quantized elementsattr.
  if (minElems || maxElems) {
    // Must have the same number of elements.
    if (minElems.getNumElements() != maxElems.getNumElements())
      return {};
    min.reserve(minElems.getNumElements());
    max.reserve(maxElems.getNumElements());
    for (auto i : minElems)
      min.push_back(FloatAttr::getValueAsDouble(i));
    for (auto i : maxElems)
      max.push_back(FloatAttr::getValueAsDouble(i));
  } else { // Just a single FP value.
    auto minVal = dyn_cast<FloatAttr>(minAttr);
    if (minVal)
      min.push_back(minVal.getValueAsDouble());
    else
      return {};
    auto maxVal = dyn_cast<FloatAttr>(maxAttr);
    if (maxVal)
      max.push_back(maxVal.getValueAsDouble());
    else
      return {};
  }

  if (min.size() == max.size()) {
    if (min.size() == 1) { // Per-tensor quantization with one min/max pair.
      retType = quant::fakeQuantAttrsToType(
          builder.getUnknownLoc(), quantBits.getInt(), min[0], max[0],
          narrowRange.getValue(), convfunc.expressedType, isSigned);
    } else if (min.size() > 1) { // Per-axis quant on filterQuantDim.
      auto shape = dyn_cast<ShapedType>(inputDType);
      if (!shape)
        return {};
      if ((filterQuantDim) >= 0 && (shape.getRank() > filterQuantDim)) {
        retType = quant::fakeQuantAttrsToType(
            builder.getUnknownLoc(), quantBits.getInt(), filterQuantDim, min[0],
            max[0], narrowRange.getValue(), convfunc.expressedType, isSigned);
      }
    } else {
      return {};
    }
  } else {
    return {};
  }

  if (!retType)
    return {};

  return convfunc.convert(retType);
}

/// Builds RVTensor quantization attributes from min/max values.
TypeAttr
mlir::rvtensor::buildQTypeAttrFromMinMax(OpBuilder builder, Type inputDtype,
                                     Attribute minAttr, Attribute maxAttr,
                                     IntegerAttr quantBits, int filterQuantDim,
                                     bool isSigned, BoolAttr narrowRange) {

  return TypeAttr::get(buildQTypeFromMinMax(builder, inputDtype, minAttr,
                                            maxAttr, quantBits, filterQuantDim,
                                            isSigned, narrowRange));
}
