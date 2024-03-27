add_mlir_dialect(RVMathOps rvmath)
add_mlir_doc(RVMathDialect RVMathDialect RVMath/ -gen-dialect-doc)
add_mlir_doc(RVMathOps RVMathOps RVMath/ -gen-op-doc)

set(LLVM_TARGET_DEFINITIONS RVMathPasses.td)
mlir_tablegen(RVMathPasses.h.inc --gen-pass-decls)
add_public_tablegen_target(MLIRRVMathPassesIncGen)
