# A RVTensor MLIR dialect

This is an out-of-tree [MLIR](https://mlir.llvm.org/) RVTensor dialect.

The RISC-V Tensor dialect, abbreviated as RVTensor, introduces a suite of whole-tensor operations frequently utilized in Deep Neural
Networks (DNNs), such as convolutions and matrix multiplications. The primary goal of RVTensor is to facilitate the conversion of these
high-level operations into vendor-specific RISC-V intrinsic operations, leveraging the capabilities of the vector and/or matrix Instruction
Set Architecture (ISA). This ensures that operators from popular Machine Learning (ML) frameworks like TensorFlow and PyTorch can be
efficiently represented within RVTensor. Additionally, it is anticipated that tools will be developed to seamlessly translate operations from
ML frameworks into RVTensor, enhancing interoperability and optimization for RISC-V based implementations.

## Building - Component Build

This setup assumes that you have built LLVM and MLIR in `$BUILD_DIR` and installed them to `$PREFIX`. To build and launch the tests, run
```sh
mkdir build && cd build
cmake -G Ninja .. -DMLIR_DIR=$PREFIX/lib/cmake/mlir -DLLVM_EXTERNAL_LIT=$BUILD_DIR/bin/llvm-lit
cmake --build . --target check-rvtensor
```
To build the documentation from the TableGen description of the dialect operations, run
```sh
cmake --build . --target mlir-doc
```
**Note**: Make sure to pass `-DLLVM_INSTALL_UTILS=ON` when building LLVM with CMake in order to install `FileCheck` to the chosen
installation prefix.

## Building - Monolithic Build

This setup assumes that you build the project as part of a monolithic LLVM build via the `LLVM_EXTERNAL_PROJECTS` mechanism.
To build LLVM, MLIR, the example and launch the tests run
```sh
mkdir build && cd build
cmake -G Ninja `$LLVM_SRC_DIR/llvm` \
    -DCMAKE_BUILD_TYPE=Release \
    -DLLVM_TARGETS_TO_BUILD=host \
    -DLLVM_ENABLE_PROJECTS=mlir \
    -DLLVM_EXTERNAL_PROJECTS=rvtensor-dialect -DLLVM_EXTERNAL_RVTENSOR_DIALECT_SOURCE_DIR=../
cmake --build . --target check-rvtensor
```
Here, `$LLVM_SRC_DIR` needs to point to the root of the monorepo.
