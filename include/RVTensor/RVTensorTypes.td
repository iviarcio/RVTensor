//===- RVTensorTypes.td - RVTensor dialect types -------*- tablegen -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef RVTENSOR_TYPES
#define RVTENSOR_TYPES

include "mlir/IR/CommonTypeConstraints.td"

//===----------------------------------------------------------------------===//
// RVTensor Non-Quantized Signed Integer Types.
//===----------------------------------------------------------------------===//

def RVTensor_UInt8  : UI<8>;
def RVTensor_UInt16 : UI<16>;
def RVTensor_UInt32 : UI<32>;

def RVTensor_Int8  : I<8>;
def RVTensor_Int16 : I<16>;
def RVTensor_Int32 : I<32>;
def RVTensor_Int64 : I<64>;

def RVTensor_SignedInt : AnyTypeOf<[RVTensor_Int8,
                                  RVTensor_Int16,
                                  RVTensor_Int32,
                                  RVTensor_Int64]>;

def RVTensor_Bool : I<1>;

// No unsigned unquantized int types.
def RVTensor_Int : AnyTypeOf<[RVTensor_Bool,
                            RVTensor_UInt8,
                            RVTensor_UInt16,
                            RVTensor_UInt32,
                            RVTensor_SignedInt]>;

def RVTensor_Int32Or64 : AnyTypeOf<[RVTensor_Int32,
                   	              RVTensor_Int64]>;

//===----------------------------------------------------------------------===//
// RVTensor Quantized Types.
//===----------------------------------------------------------------------===//

// Base class of a quantized type.
// Param tuple is: [bitwidth, qpoint, smantissa, sexp, low_end, high_end].
// Low and high ends: 0,255 when unsigned, -128,127 when signed
class RVTensor_QuantizedType<string n, list<int> params, bit signed>
  : Type<And<[CPred<"::llvm::isa<mlir::quant::QuantizedType>($_self)">,
              CPred<"::llvm::cast<mlir::quant::QuantizedType>($_self)" #
                    ".getStorageTypeIntegralWidth() == " # !head(params)>]>,
    "Q" # !if (signed, "int", "uint") # !head(params) # " type"> {
  string name = n;
  string asTraitArgsStr = !interleave(params, ", ") #
                          !if(signed, ", true", ", false");
}

def RVTensor_QuantizedInt	: AnyTypeOf<[RVTensor_QuantizedType<"uint8", [8], 0>,
                                     RVTensor_QuantizedType<"int8", [8, 0], 1>,
                                     RVTensor_QuantizedType<"int16", [16, 0], 1>,
                                     RVTensor_QuantizedType<"int32", [32, 0], 1>]>;

//===----------------------------------------------------------------------===//
// RVTensor Floating-point types.
//===----------------------------------------------------------------------===//

def RVTensor_Float : AnyTypeOf<[F16, F32]>;

//===----------------------------------------------------------------------===//
// RVTensor Multi-category types.
//===----------------------------------------------------------------------===//

def RVTensor_AnyNumber : AnyTypeOf<[RVTensor_Int, RVTensor_QuantizedInt, RVTensor_Float],
                                "number">;

// For weight tensors from rvtensor::Conv2DOp, rvtensor::Conv3DOp,
// rvtensor::DepthwiseConv2DOp, rvtensor::TransposeConv2DOp, rvtensor::FullyConnectedOp
def RVTensor_Weight : AnyTypeOf<[RVTensor_Int8, RVTensor_QuantizedInt, RVTensor_Float]>;

//===----------------------------------------------------------------------===//
// RVTensor Tensor types
//===----------------------------------------------------------------------===//

def RVTensor_Int32Tensor : TensorOf<[RVTensor_Int32]>;
def RVTensor_Int32Or64Tensor : TensorOf<[RVTensor_Int32Or64]>;

def RVTensor_FloatTensor : TensorOf<[RVTensor_Float]>;

// Either ranked or unranked tensor of rvtensor supported element types.
def RVTensor_Tensor : TensorOf<[RVTensor_AnyNumber]>;

// Must be ranked but no further constraints
def RVTensor_RankedTensor : RankedTensorOf<[RVTensor_AnyNumber]>;

// Any tensor element type allowed in RVTensor ops.
def RVTensor_ElementType : Type<Or<[RVTensor_Int.predicate, RVTensor_QuantizedInt.predicate,
                                RVTensor_Float.predicate]>, "rvtensor.dtype">;

class RVTensor_TensorOfOrNone<list<Type> allowedTypes, string description = ""> :
  AnyTypeOf<[TensorOf<allowedTypes>, NoneType], description>;

// Tensor types with constrained ranks.
def RVTensor_UnrankedTensor : UnrankedTensorOf<[RVTensor_AnyNumber]>;

def RVTensor_Tensor1D : AnyTypeOf<[RVTensor_UnrankedTensor, 1DTensorOf<[RVTensor_AnyNumber]>]>;
def RVTensor_Tensor2D : AnyTypeOf<[RVTensor_UnrankedTensor, 2DTensorOf<[RVTensor_AnyNumber]>]>;
def RVTensor_Tensor3D : AnyTypeOf<[RVTensor_UnrankedTensor, 3DTensorOf<[RVTensor_AnyNumber]>]>;
def RVTensor_Tensor4D : AnyTypeOf<[RVTensor_UnrankedTensor, 4DTensorOf<[RVTensor_AnyNumber]>]>;
def RVTensor_Tensor5D : AnyTypeOf<[RVTensor_UnrankedTensor, TensorRankOf<[RVTensor_AnyNumber], [5]>]>;

//===----------------------------------------------------------------------===//
// RVTensor Attribute predicates and classes.
//===----------------------------------------------------------------------===//

class DenseArrayMaxCt<int n> : AttrConstraint<
    CPred<"::llvm::cast<::mlir::DenseArrayAttr>($_self).size() <= " # n>,
    "with at least " # n # " elements">;

def RVTensor_IntArrayAttr2 : ConfinedAttr<DenseI64ArrayAttr, [DenseArrayCount<2>]>;
def RVTensor_IntArrayAttr3 : ConfinedAttr<DenseI64ArrayAttr, [DenseArrayCount<3>]>;
def RVTensor_IntArrayAttr4 : ConfinedAttr<DenseI64ArrayAttr, [DenseArrayCount<4>]>;
def RVTensor_IntArrayAttr5 : ConfinedAttr<DenseI64ArrayAttr, [DenseArrayCount<5>]>;
def RVTensor_IntArrayAttr6 : ConfinedAttr<DenseI64ArrayAttr, [DenseArrayCount<6>]>;

def RVTensor_IntArrayAttrUpto2 : ConfinedAttr<DenseI64ArrayAttr, [DenseArrayMaxCt<2>]>;
def RVTensor_IntArrayAttrUpto4 : ConfinedAttr<DenseI64ArrayAttr, [DenseArrayMaxCt<4>]>;
def RVTensor_IntArrayAttrUpto5 : ConfinedAttr<DenseI64ArrayAttr, [DenseArrayMaxCt<5>]>;

//===----------------------------------------------------------------------===//
// RVTensor Generic scalar, vector, or tensor of a particular type.
//===----------------------------------------------------------------------===//

class RVTensor_TypeLike<list<Type> types, string description = ""> : TypeConstraint<Or<[
     AnyTypeOf<types>.predicate,
     VectorOf<types>.predicate,
     TensorOf<types>.predicate]>,
     description>;

def RVTensor_IntLike   : RVTensor_TypeLike<[RVTensor_Int], "integer-like">;
def RVTensor_Int8Like  : RVTensor_TypeLike<[RVTensor_Int8], "integer-8-bit-like">;
def RVTensor_Int16Like : RVTensor_TypeLike<[RVTensor_Int16], "integer-16-bit-like">;
def RVTensor_Int32Like : RVTensor_TypeLike<[RVTensor_Int32], "integer-32-bit-like">;
def RVTensor_Int64Like : RVTensor_TypeLike<[RVTensor_Int64], "integer-64-bit-like">;

#endif // RVTENSOR_TYPES
