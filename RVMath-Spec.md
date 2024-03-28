== RVMath Specification

The RISC-V Mathematics dialect (RVMath) defines a set of primitive operators to which higher level operators can be lowered in a consistent
way. To remain effective and efficient to implement, the set of operators must be constrained to a reasonably small set of primitive
operations out of which others can be constructed.

=== Goals

The goals of RVMath include the following:

* A minimal and stable set of tensor-level operators to which machine learning framework operators can be reduced.

* Full support for both quantized integer and floating-point content.

* Precise functional description of the behavior of each operator, including the treatment of their numerical behavior in the case of precision,
scaling, and range as required by quantized datatypes.

* Agnostic to any single high-level framework, compiler backend stack or particular Risc-V vendor.

=== Status

The RVMath specification is a work in progress.

=== Tensor Definitions

Tensors are essentially multidimensional data arrays. Accompanying these tensors is metadata that outlines their characteristics, which
include:

- **Data Type and Shape**: The term "rank" refers to the number of dimensions a tensor possesses. A tensor can have a rank of zero, in
which case it is termed a scalar and contains a single value. The "shape" of a tensor, an integer array, has a length equal to the tensor's
rank. Each integer in this array signifies the count of elements in its corresponding dimension. The requirement is that the shape in every
dimension must be one or more. For tensors with a non-zero rank, a specific type called `shape_t` is used to represent their shape.
`shape_t` is a one-dimensional array whose length matches the tensor's rank, with each component being of type `size_t`, representing the
dimension's size.

- **Size Limitations**: The overall size of a tensor is bounded by the `size_t` data type. `size_t` must be capable of representing
integers within the range from 0 to (1 << (MAX_LOG2_SIZE + 1)) - 1, where `MAX_LOG2_SIZE` is determined by the system's architecture
(usually 63 for 64-bit systems and 31 for 32-bit systems). The total size of a tensor, calculated as the number of elements times the size
of each element in bytes (assumed to be 1 for elements smaller than 8 bits), cannot exceed (1 << (MAX_LOG2_SIZE + 1)) - 1.

- **Dimensional Size Constraints**: The size of a tensor in each dimension is also limited by the `size_t` type, setting the maximum size
of a tensor in any dimension to (1 << MAX_LOG2_SIZE) - 1, and thus the highest possible index to (1 << MAX_LOG2_SIZE) - 2. All tensor
indices must be non-negative.

- **Data Layouts**: The following data layouts are supported in RVMath.

|Name|Description of dimensions|Usage|
|----|-------------------------|-----|
|NCHW|Batch, Channels, Height, Width|Feature maps|
|NDCHW|Batch, Depth, Channels, Height, Width|Feature maps for 3D convolution|
|OIHW|Output channels, Input channels, Filter Height, Filter Width|Weights|
|IMHW|Input channels, Channel Multiplier, Filter Height, Filter Width|Weights for depthwise convolutions|
|DOIHW|Depth, Output Channels, Input Channels, Filter Height, Filter Width|Weights for 3D convolution|

=== Quantization

In Machine Learning frameworks, tensors can be implemented in a quantized format, where integer values are used to approximate the original
floating-point numbers. Unlike some operations that might automatically adjust scales to accommodate quantized values, RVMath's integer
operations require explicit handling. Quantization points, or q-point values, must be explicitly provided to each operator, ensuring
accurate processing in line with each operator's specific requirements.

To transition a network with quantized tensors to RVMath, it's necessary to employ explicit RESCALE operators wherever there's a change in
quantization scale. This approach simplifies quantized operations to integer-only computations. The RESCALE operation in RVMath is crucial
for adjusting values across different precisions. Defined through a combination of integer multiplication, addition, and bit shifting, this
operator allows for precise scaling adjustments. RVMath accommodates two multiplier precisions: 16-bit and 32-bit, ensuring calculations
stay within the bounds of a 64-bit accumulator and that the end result remains within a 32-bit range. This design choice prevents overflow
and ensures the final outcome is accurately represented within the system's computational limits.

For convolution operators, scaling of the input is not necessary. The integer implementations of these operators will deduct the q-point
from the integer values as specified for each operator. The convolution process results in an accumulator output of the int32_t type. This
output is then adjusted to the desired final output range using the RESCALE operator. The scaling within the RESCALE operator is determined
by setting multiplier and shift values such that: multiplier * 2^(-shift) = (input scale * weight scale) / output scale. Here, the input
scale, weight scale, and output scale refer to the conversion factors from integer to floating-point for the input, weight, and output
tensor values, respectively. If scaling on a per-channel basis is required, then the per-channel version of the RESCALE operation should be
utilized.

When operations involve two quantized tensors, they must represent the same numerical range for the results to be considered valid. In such
scenarios, RVMath anticipates the use of RESCALE operators as needed to align 32-bit integer values within a common range. There is a
variety of valid choices for scale factors and the selection of a common range. RVMath does not prescribe specific scale factors and ranges
that must be used. Compilers generating RVMath sequences are encouraged to select a range that prevents overflow and maximizes the accuracy
of the output.

=== RVMath Operators

==== CONV2D

Performs a 2D convolution over the given tensor input, using the weight tensor.

==== CONV3D

Performs a 3D convolution over the given input tensor.

==== DEPTHWISE_CONV2D

Performs 2D convolutions separately over each channel of the given tensor input, using the weight tensor.

==== FULLY_CONNECTED

Performs a fully connected network.

==== MATMUL

==== TRANSPOSE_CONV2D

Performs a 2D transposed convolution over the given tensor input, using the weights tensor.

==== RESCALE

Rescale quantized values into a new domain. This function scales by factor: multiplier * 2^-shift^.

==== RESHAPE

Returns a tensor with the same type/values as the input, with a new shape specified by the shape argument. Reshape may operate on tensors of any rank. No data conversion happens during a reshape operation.

==== SLICE

Extracts a slice of input1, beginning at the start coordinates, and extending for size elements in each direction.
No data conversion happens during a slice operation.

==== TILE

Replicates input1 multiples times along each dimension.

==== TRANSPOSE

Permutes the dimensions of the input tensor input1 based on the perms argument.
Each value in the perms list must be a valid dimension of the input tensor and may not be repeated.

