add_mlir_dialect(RVTensorOps rvtensor)
add_mlir_doc(RVTensorDialect RVTensorDialect RVTensor/ -gen-dialect-doc)
add_mlir_doc(RVTensorOps RVTensorOps RVTensor/ -gen-op-doc)

set(LLVM_TARGET_DEFINITIONS RVTensorPasses.td)
mlir_tablegen(RVTensorPasses.h.inc --gen-pass-decls)
add_public_tablegen_target(RVTensorPassesIncGen)

add_dependencies(mlir-headers RVTensorPassesIncGen)
add_mlir_doc(RVTensorPasses RVTensorPasses ./ -gen-pass-doc)

set(LLVM_TARGET_DEFINITIONS RVTensorOps.td)
mlir_tablegen(RVTensorAttributes.h.inc -gen-attrdef-decls)
mlir_tablegen(RVTensorAttributes.cpp.inc -gen-attrdef-defs)
add_public_tablegen_target(RVTensorAttributesIncGen)

set(LLVM_TARGET_DEFINITIONS RVTensorDialect.td)
mlir_tablegen(RVTensorDialect.cpp.inc -gen-bytecode -bytecode-dialect="RVTensor")
add_public_tablegen_target(RVTensorDialectIncGen)
