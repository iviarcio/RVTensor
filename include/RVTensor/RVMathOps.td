//===- RVTensorOps.td - RVTensor dialect ops -----------*- tablegen -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef RVTENSOR_OPS
#define RVTENSOR_OPS

include "RVTensor/RVTensorTypes.td"
include "RVTensor/RVTensorDialect.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

//===----------------------------------------------------------------------===//
// RVTensor Operator: conv2d
//===----------------------------------------------------------------------===//

def RVTensor_Conv2DOp : RVTensor_InferShapedTypeOp<"conv2d"> {
  let summary = "2D Convolution Operator";

  let description = [{
    Performs a 2D convolution over the given tensor input, using the weight
    tensor.
  }];

  let arguments = (ins
    RVTensor_Tensor4D:$input,
    4DTensorOf<[RVTensor_Weight]>:$weight,
    RVTensor_Tensor1D:$bias,
    RVTensor_IntArrayAttr4:$pad,
    RVTensor_IntArrayAttr2:$stride,
    RVTensor_IntArrayAttr2:$dilation,
    OptionalAttr<RVTensor_ConvOpQuantizationAttr>:$quantization_info
  );

  let results = (outs
    RVTensor_Tensor4D:$output
  );

  let builders = [RVTensor_ConvOpQuantInfoBuilder];
  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// RVTensor Operator: conv3d
//===----------------------------------------------------------------------===//
def RVTensor_Conv3DOp : RVTensor_InferShapedTypeOp<"conv3d"> {
  let summary = "3D Convolution operator";

  let description = [{
    Performs a 3D convolution over the given input tensor.
  }];

  let arguments = (ins
    RVTensor_Tensor5D:$input,
    TensorRankOf<[RVTensor_Weight], [5]>:$weight,
    RVTensor_Tensor1D:$bias,
    RVTensor_IntArrayAttr6:$pad,
    RVTensor_IntArrayAttr3:$stride,
    RVTensor_IntArrayAttr3:$dilation,
    OptionalAttr<RVTensor_ConvOpQuantizationAttr>:$quantization_info
  );

  let results = (outs
    RVTensor_Tensor5D:$output
  );

  let builders = [RVTensor_ConvOpQuantInfoBuilder];
  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// RVTensor Operator: depthwise_conv2d
//===----------------------------------------------------------------------===//
def RVTensor_DepthwiseConv2DOp : RVTensor_InferShapedTypeOp<"depthwise_conv2d"> {
  let summary = "Depthwise 2D Convolution operator";

  let description = [{
    Performs 2D convolutions separately over each channel of the given tensor
    input, using the weight tensor.
  }];

  let arguments = (ins
    RVTensor_Tensor4D:$input,
    4DTensorOf<[RVTensor_Weight]>:$weight,
    RVTensor_Tensor1D:$bias,
    RVTensor_IntArrayAttr4:$pad,
    RVTensor_IntArrayAttr2:$stride,
    RVTensor_IntArrayAttr2:$dilation,
    OptionalAttr<RVTensor_ConvOpQuantizationAttr>:$quantization_info
  );

  let results = (outs
    RVTensor_Tensor4D:$output
  );

  let builders = [RVTensor_ConvOpQuantInfoBuilder];
  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// RVTensor Operator: transpose_conv2d
//===----------------------------------------------------------------------===//

def RVTensor_TransposeConv2DOp : RVTensor_InferShapedTypeOp<"transpose_conv2d"> {
  let summary = "Transpose 2D Convolution operator.";

  let description = [{
    Performs a 2D transposed convolution over the given tensor input, using the
    weights tensor.
  }];

  let arguments = (ins
    RVTensor_Tensor4D:$input,
    4DTensorOf<[RVTensor_Weight]>:$filter,
    RVTensor_Tensor1D:$bias,
    RVTensor_IntArrayAttr4:$out_pad,
    RVTensor_IntArrayAttr2:$stride,
    RVTensor_IntArrayAttrUpto4:$out_shape,
    OptionalAttr<RVTensor_ConvOpQuantizationAttr>:$quantization_info
  );

  let results = (outs
    RVTensor_Tensor4D:$output
  );

  let builders = [RVTensor_TransConvOpQuantInfoBuilder];
}

//===----------------------------------------------------------------------===//
// RVTensor Operator: fully_connected
//===----------------------------------------------------------------------===//
def RVTensor_FullyConnectedOp : RVTensor_InferShapedTypeOp<"fully_connected"> {
  let summary = "Fully Connected operator";

  let description = [{
    Performs a fully connected network.
  }];

  let arguments = (ins
    RVTensor_Tensor2D:$input,
    2DTensorOf<[RVTensor_Weight]>:$weight,
    RVTensor_Tensor1D:$bias,
    OptionalAttr<RVTensor_ConvOpQuantizationAttr>:$quantization_info
  );

  let results = (outs
    RVTensor_Tensor2D:$output
  );

  let builders = [RVTensor_FCOpQuantInfoBuilder];
  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// RVTensor Operator: matmul
//===----------------------------------------------------------------------===//
def RVTensor_MatMulOp : RVTensor_InferShapedTypeOp<"matmul"> {
  let summary = "Matrix multiplication with bias";

  let description = [{
    Performs a two dimensional matrix multiplication. This allows both inputs to
    be activations, rather than reserving weights as an attribute in the
    FULLY_CONNECTED operator.
  }];

  let arguments = (ins
    RVTensor_Tensor3D:$a,
    RVTensor_Tensor3D:$b,
    OptionalAttr<RVTensor_MatMulOpQuantizationAttr>:$quantization_info
  );

  let results = (outs
    RVTensor_Tensor3D:$c
  );

  let builders = [RVTensor_MatMulOpQuantInfoBuilder];
}

//===----------------------------------------------------------------------===//
// RVTensor Operator: rescale
//===----------------------------------------------------------------------===//

def RVTensor_RescaleOp: RVTensor_Op<"rescale", [Pure,
      DeclareOpInterfaceMethods<InferShapedTypeOpInterface,
                              ["inferReturnTypeComponents"]>]> {
  let summary = "RVTensor rescale operator";

  let description = [{
    Rescale quantized values into a new domain. Supported rescalings are:
    Mode                    Input   Output
    signed 8 to 8           int8    int8
    signed 8 to 16          int8    int16
    signed 8 to 32          int8    int32
    signed 16 to 8          int16   int8
    signed 16 to 16         int16   int16
    signed 16 to 32         int16   int32
    signed 32 to 8          int32   int8
    signed 32 to 16         int32   int16
    signed 32 to 32         int32   int32
    unsigned 8 to signed 8  uint8   int8
    signed 8 to unsigned 8  int8    uint8
  }];

  let arguments = (ins
    RVTensor_Tensor:$input,
    I32Attr:$in_qpoint,
    I32Attr:$out_qpoint,
    DenseI32ArrayAttr:$multiplier,
    DenseI8ArrayAttr:$shift,
    BoolAttr:$scale32,
    BoolAttr:$per_channel
  );

  let results = (outs
    RVTensor_Tensor:$output
  );

  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

//===----------------------------------------------------------------------===//
// RVTensor Operator: reshape
//===----------------------------------------------------------------------===//
def RVTensor_ReshapeOp : RVTensor_InferTensorTypeOp<"reshape"> {
  let summary = "Reshape operator";

  let description = [{
    Returns a tensor with the same type/values as the input, with a new shape
    specified by the shape argument. Reshape may operate on tensors of any rank.
    No data conversion happens during a reshape operation.
  }];

  let hasFolder = 1;
  let hasVerifier = 1;

  let arguments = (ins
    RVTensor_Tensor:$input1,
    DenseI64ArrayAttr:$new_shape
  );

  let results = (outs
    RVTensor_RankedTensor:$output
  );

  let extraClassDeclaration = [{
    /// Returns true when two result types are compatible for this op;
    /// Method used by InferTypeOpInterface.
    static bool isCompatibleReturnTypes(TypeRange l, TypeRange r);
  }];

  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

//===----------------------------------------------------------------------===//
// RVTensor Operator: transpose
//===----------------------------------------------------------------------===//
def RVTensor_TransposeOp : RVTensor_InferShapedTypeOp<"transpose"> {
  let summary = "Transpose operator";

  let description = [{
    Permutes the dimensions based on perm.
  }];

  let arguments = (ins
    RVTensor_Tensor:$input1,
    RVTensor_Int32Or64Tensor:$perms
  );

  let results = (
    outs RVTensor_Tensor:$output
  );

  let extraClassDeclaration = [{
    LogicalResult getConstantPerms(llvm::SmallVector<int64_t> &perms);
  }];

  let hasCanonicalizer = 1;
  let hasFolder = 1;
  let hasVerifier = 1;
}

#endif // RVTENSOR_OPS
