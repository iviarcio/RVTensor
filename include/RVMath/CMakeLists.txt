add_mlir_dialect(RVTensorOps rvtensor)
add_mlir_doc(RVTensorDialect RVTensorDialect RVTensor/ -gen-dialect-doc)
add_mlir_doc(RVTensorOps RVTensorOps RVTensor/ -gen-op-doc)

set(LLVM_TARGET_DEFINITIONS RVTensorPasses.td)
mlir_tablegen(RVTensorPasses.h.inc --gen-pass-decls)
add_public_tablegen_target(MLIRRVTensorPassesIncGen)
