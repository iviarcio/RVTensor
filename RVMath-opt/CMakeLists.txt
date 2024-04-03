get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)
set(LIBS
        ${dialect_libs}
        ${conversion_libs}
        MLIRArithDialect
        MLIROptLib
        MLIRRVTensor
        )
add_llvm_executable(rvtensor-opt rvtensor-opt.cpp)

llvm_update_compile_flags(rvtensor-opt)
target_link_libraries(rvtensor-opt PRIVATE ${LIBS})

mlir_check_all_link_libraries(rvtensor-opt)
