get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)
set(LIBS
        ${dialect_libs}
        ${conversion_libs}
        MLIRArithDialect
        MLIROptLib
        MLIRRVMath
        )
add_llvm_executable(rvmath-opt rvmath-opt.cpp)

llvm_update_compile_flags(rvmath-opt)
target_link_libraries(rvmath-opt PRIVATE ${LIBS})

mlir_check_all_link_libraries(rvmath-opt)
