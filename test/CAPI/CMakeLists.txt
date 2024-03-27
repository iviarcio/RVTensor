# Here we create a single aggregate shared library with the parts of the CAPI
# that we want to bundle together. Then we link a simple C executable
# against it to demonstrate that it does have the fully self contained
# core MLIR library and our own rvmath dialect.
add_mlir_aggregate(RVMathCAPITestLib
  SHARED
  EMBED_LIBS
  MLIRCAPIIR
  # TODO: Remove this in favor of showing fine grained dialect registration
  # (once available).
  MLIRCAPIRegisterEverything
  RVMathCAPI
)

add_llvm_executable(rvmath-capi-test
  rvmath-capi-test.c
)
llvm_update_compile_flags(rvmath-capi-test)
target_link_libraries(rvmath-capi-test
  PRIVATE RVMathCAPITestLib)
