//===- RVMathDialect.td - RVMath dialect -----------*- tablegen -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef RVMATH_DIALECT
#define RVMATH_DIALECT

include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/BytecodeBase.td"
include "mlir/IR/OpBase.td"

//===----------------------------------------------------------------------===//
// RVMath dialect definition.
//===----------------------------------------------------------------------===//

def RVMath_Dialect : Dialect {
    let name = "rvmath";
    let summary = "A rvmath out-of-tree MLIR dialect.";
    let description = [{
        This dialect is an out-of-tree MLIR RVMath dialect designed to operate
        the Convolutions & Matrix Multiplications on The Risc-V architecture.
    }];
    let cppNamespace = "::mlir::rvmath";

    let useDefaultTypePrinterParser = 1;
    let extraClassDeclaration = [{ void registerTypes(); }];
    let dependentDialects = ["tensor::TensorDialect", "quant::QuantizationDialect"];
    let hasConstantMaterializer = 1;
}

//===----------------------------------------------------------------------===//
// RVMath Attributes.
//===----------------------------------------------------------------------===//

class RVMath_Attr<string attrName, string attrMnemonic, list<Trait> traits = []>
    : AttrDef<RVMath_Dialect, attrName, traits> {
  let mnemonic = attrMnemonic;
}

//===----------------------------------------------------------------------===//
// RVMath bytecode reader/writer definition.
//===----------------------------------------------------------------------===//

def RVMathDialectTypes : DialectTypes<"RVMath"> {
  let elems = [];
}

//===----------------------------------------------------------------------===//
// RVMath dialect op interface.
//===----------------------------------------------------------------------===//

def RVMathOpInterface : OpInterface<"RVMathOp"> {
  let description = [{
    Implemented by ops that correspond to the RVMath specification.
  }];
}

//===----------------------------------------------------------------------===//
// RVMath Base operation class definition.
//===----------------------------------------------------------------------===//

class RVMath_Op<string mnemonic, list<Trait> traits = []> :
    Op<RVMath_Dialect, mnemonic, !listconcat(traits, [RVMathOpInterface])> {
}

class RVMath_InferTensorTypeOp<string mnemonic, list<Trait> traits = []>
    : RVMath_Op<mnemonic, !listconcat(traits, [InferTensorTypeAdaptor, Pure])> {
  let assemblyFormat =
      "operands attr-dict `:` functional-type(operands, results)";
}

class RVMath_InferShapedTypeOp<string mnemonic, list<Trait> traits = []>
    : RVMath_Op<mnemonic, !listconcat(traits, [InferShapedTypeOpAdaptor, Pure])> {
  let assemblyFormat =
      "operands attr-dict `:` functional-type(operands, results)";
}

//===----------------------------------------------------------------------===//
// RVMath Operator Quantization Attributes.
//===----------------------------------------------------------------------===//

// Quantization attributes used across RVMath operators. Quantization attributes
// feed numerical precision parameters to the functional implementation of RVMath
// operators.

// These quantization attributes hold input and weight q-point.

// Both the ConvOp and MatMulOp QuantizationAttrs follow a common design semantic
// where their own quantization attribute only expresses the numerical behavior at
// the inputs. The scaling of their accumulator output is done using an explicit
// rvmath.rescale operator that scales the accumulator result to output scale.

def RVMath_ConvOpQuantizationAttr
    : RVMath_Attr<"ConvOpQuantization", "conv_quant"> {
  let summary = "Attribute for Conv type op quantization information.";
  let parameters = (ins "int64_t":$in_qpoint, "int64_t":$w_qpoint);
  let assemblyFormat = "`<` struct(params) `>`";
}

def RVMath_MatMulOpQuantizationAttr
    : RVMath_Attr< "MatMulOpQuantization", "matmul_quant"> {
  let summary = "Attribute for MatMulOp quantization information.";
  let parameters = (ins "int64_t":$a_qpoint, "int64_t":$b_qpoint);
  let assemblyFormat = "`<` struct(params) `>`";
}

//===----------------------------------------------------------------------===//
// RVMath Operator Quantization Builders.
//===----------------------------------------------------------------------===//

// This builder is called on all convolution operators except for TransposeConv,
// which has specialized output shape semantics. The builder also defines the
// bitwidth of the output given the bit width of the input & weight content.

def RVMath_ConvOpQuantInfoBuilder : OpBuilder<
  (ins "::mlir::Type":$outputType, "::mlir::Value":$input,
       "::mlir::Value":$weight, "::mlir::Value":$bias,
       "::mlir::DenseI64ArrayAttr":$pad, "::mlir::DenseI64ArrayAttr":$stride,
       "::mlir::DenseI64ArrayAttr":$dilation),
  [{
    buildConvOpWithQuantInfo($_builder, $_state, outputType,
                             input, weight, bias,
                             pad, stride, dilation);
  }]>;

// Handles rvmath.transpose_conv2d which has an outpad and output shape attribute

def RVMath_TransConvOpQuantInfoBuilder : OpBuilder<
  (ins "::mlir::Type":$outputType, "::mlir::Value":$input,
       "::mlir::Value":$weight, "mlir::Value":$bias,
       "::mlir::DenseI64ArrayAttr":$outpad,
       "::mlir::DenseI64ArrayAttr":$stride,
       "::mlir::DenseI64ArrayAttr":$outputShape),
  [{
    buildTransConvOpWithQuantInfo($_builder, $_state, outputType,
                                  input, weight, bias,
                                  outpad, stride,
                                  outputShape);
  }]>;

// The rvmath.fully_connected op has its own builder as it does not have
// strides, dilation, and padding.

def RVMath_FCOpQuantInfoBuilder : OpBuilder<
  (ins "Type":$outputType, "Value":$input, "Value":$weight, "Value":$bias),
  [{
    buildFCOpWithQuantInfo($_builder, $_state, outputType,
                           input, weight, bias);
  }]>;

// The rvmath.matmul op is also intended to be generated where a fully_connected
// op must be constructed where the weight is not a constant. In this case,
// the fully_connected op must be expressed using matmul.

def RVMath_MatMulOpQuantInfoBuilder : OpBuilder<
  (ins "Type":$outputType, "Value":$a, "Value":$b),
  [{
    buildMatMulOpWithQuantInfo($_builder, $_state, outputType, a, b);
  }]>;

#endif // RVMATH_DIALECT
