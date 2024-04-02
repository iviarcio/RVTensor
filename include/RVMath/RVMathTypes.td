//===- RVMathTypes.td - RVMath dialect types -------*- tablegen -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef RVMATH_TYPES
#define RVMATH_TYPES

include "mlir/IR/OpBase.td"
include "mlir/IR/AttrTypeBase.td"
include "RVMath/RVMathDialect.td"

//===----------------------------------------------------------------------===//
// RVMath Non-Quantized Signed Integer Types.
//===----------------------------------------------------------------------===//

def RVMath_UInt8  : UI<8>;
def RVMath_UInt16 : UI<16>;
def RVMath_UInt32 : UI<32>;

def RVMath_Int8  : I<8>;
def RVMath_Int16 : I<16>;
def RVMath_Int32 : I<32>;
def RVMath_Int64 : I<64>;

def RVMath_SignedInt : AnyTypeOf<[RVMath_Int8,
                                  RVMath_Int16,
                                  RVMath_Int32,
                                  RVMath_Int64]>;

def RVMath_Bool : I<1>;

// No unsigned unquantized int types.
def RVMath_Int : AnyTypeOf<[RVMath_Bool,
                            RVMath_UInt8,
                            RVMath_UInt16,
                            RVMath_UInt32,
                            RVMath_SignedInt]>;

def RVMath_Int32Or64 : AnyTypeOf<[RVMath_Int32,
                   	              RVMath_Int64]>;

//===----------------------------------------------------------------------===//
// RVMath Quantized Types.
//===----------------------------------------------------------------------===//

// Base class of a quantized type.
// Param tuple is: [bitwidth, qpoint, smantissa, sexp, low_end, high_end].
// Low and high ends: 0,255 when unsigned, -128,127 when signed
class RVMath_QuantizedType<string n, list<int> params, bit signed>
  : Type<And<[CPred<"::llvm::isa<mlir::quant::QuantizedType>($_self)">,
              CPred<"::llvm::cast<mlir::quant::QuantizedType>($_self)" #
                    ".getStorageTypeIntegralWidth() == " # !head(params)>]>,
    "Q" # !if (signed, "int", "uint") # !head(params) # " type"> {
  string name = n;
  string asTraitArgsStr = !interleave(params, ", ") #
                          !if(signed, ", true", ", false");
}

def RVMath_QuantizedInt	: AnyTypeOf<[RVMath_QuantizedType<"uint8", [8], 0>,
                                     RVMath_QuantizedType<"int8", [8, 0], 1>,
                                     RVMath_QuantizedType<"int16", [16, 0], 1>,
                                     RVMath_QuantizedType<"int32", [32, 0], 1>]>;

//===----------------------------------------------------------------------===//
// RVMath Floating-point types.
//===----------------------------------------------------------------------===//

def RVMath_Float : AnyTypeOf<[F16, F32]>;

//===----------------------------------------------------------------------===//
// RVMath Multi-category types.
//===----------------------------------------------------------------------===//

def RVMath_AnyNumber : AnyTypeOf<[RVMath_Int, RVMath_QuantizedInt, RVMath_Float],
                                "number">;

// For weight tensors from rvmath::Conv2DOp, rvmath::Conv3DOp,
// rvmath::DepthwiseConv2DOp, rvmath::TransposeConv2DOp, rvmath::FullyConnectedOp
def RVMath_Weight : AnyTypeOf<[RVMath_Int8, RVMath_QuantizedInt, RVMath_Float]>;

//===----------------------------------------------------------------------===//
// RVMath Tensor types
//===----------------------------------------------------------------------===//

def RVMath_Int32Tensor : TensorOf<[RVMath_Int32]>;
def RVMath_Int32Or64Tensor : TensorOf<[RVMath_Int32Or64]>;

def RVMath_FloatTensor : TensorOf<[RVMath_Float]>;

// Either ranked or unranked tensor of rvmath supported element types.
def RVMath_Tensor : TensorOf<[RVMath_AnyNumber]>;

// Must be ranked but no further constraints
def RVMath_RankedTensor : RankedTensorOf<[RVMath_AnyNumber]>;

// Any tensor element type allowed in RVMath ops.
def RVMath_ElementType : Type<Or<[RVMath_Int.predicate, RVMath_QuantizedInt.predicate,
                                RVMath_Float.predicate]>, "rvmath.dtype">;

class RVMath_TensorOfOrNone<list<Type> allowedTypes, string description = ""> :
  AnyTypeOf<[TensorOf<allowedTypes>, NoneType], description>;

// Tensor types with constrained ranks.
def RVMath_UnrankedTensor : UnrankedTensorOf<[RVMath_AnyNumber]>;

def RVMath_Tensor1D : AnyTypeOf<[RVMath_UnrankedTensor, 1DTensorOf<[RVMath_AnyNumber]>]>;
def RVMath_Tensor2D : AnyTypeOf<[RVMath_UnrankedTensor, 2DTensorOf<[RVMath_AnyNumber]>]>;
def RVMath_Tensor3D : AnyTypeOf<[RVMath_UnrankedTensor, 3DTensorOf<[RVMath_AnyNumber]>]>;
def RVMath_Tensor4D : AnyTypeOf<[RVMath_UnrankedTensor, 4DTensorOf<[RVMath_AnyNumber]>]>;
def RVMath_Tensor5D : AnyTypeOf<[RVMath_UnrankedTensor, TensorRankOf<[RVMath_AnyNumber], [5]>]>;

//===----------------------------------------------------------------------===//
// RVMath Attribute predicates and classes.
//===----------------------------------------------------------------------===//

class DenseArrayMaxCt<int n> : AttrConstraint<
    CPred<"::llvm::cast<::mlir::DenseArrayAttr>($_self).size() <= " # n>,
    "with at least " # n # " elements">;

def RVMath_IntArrayAttr2 : ConfinedAttr<DenseI64ArrayAttr, [DenseArrayCount<2>]>;
def RVMath_IntArrayAttr3 : ConfinedAttr<DenseI64ArrayAttr, [DenseArrayCount<3>]>;
def RVMath_IntArrayAttr4 : ConfinedAttr<DenseI64ArrayAttr, [DenseArrayCount<4>]>;
def RVMath_IntArrayAttr5 : ConfinedAttr<DenseI64ArrayAttr, [DenseArrayCount<5>]>;
def RVMath_IntArrayAttr6 : ConfinedAttr<DenseI64ArrayAttr, [DenseArrayCount<6>]>;

def RVMath_IntArrayAttrUpto2 : ConfinedAttr<DenseI64ArrayAttr, [DenseArrayMaxCt<2>]>;
def RVMath_IntArrayAttrUpto4 : ConfinedAttr<DenseI64ArrayAttr, [DenseArrayMaxCt<4>]>;
def RVMath_IntArrayAttrUpto5 : ConfinedAttr<DenseI64ArrayAttr, [DenseArrayMaxCt<5>]>;

//===----------------------------------------------------------------------===//
// RVMath Generic scalar, vector, or tensor of a particular type.
//===----------------------------------------------------------------------===//

class RVMath_TypeLike<list<Type> types, string description = ""> : TypeConstraint<Or<[
     AnyTypeOf<types>.predicate,
     VectorOf<types>.predicate,
     TensorOf<types>.predicate]>,
     description>;

def RVMath_IntLike   : RVMath_TypeLike<[RVMath_Int], "integer-like">;
def RVMath_Int8Like  : RVMath_TypeLike<[RVMath_Int8], "integer-8-bit-like">;
def RVMath_Int16Like : RVMath_TypeLike<[RVMath_Int16], "integer-16-bit-like">;
def RVMath_Int32Like : RVMath_TypeLike<[RVMath_Int32], "integer-32-bit-like">;
def RVMath_Int64Like : RVMath_TypeLike<[RVMath_Int64], "integer-64-bit-like">;

#endif // RVMATH_TYPES
