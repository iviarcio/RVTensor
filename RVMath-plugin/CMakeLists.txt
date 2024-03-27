add_llvm_library(RVMathPlugin
        # BUILDTREE_ONLY is only for testing purposes
        MODULE BUILDTREE_ONLY
        rvmath-plugin.cpp

        DEPENDS
        MLIRRVMath
        PLUGIN_TOOL
        mlir-opt

        LINK_LIBS
        MLIRRVMath
        )
target_include_directories(
        RVMathPlugin
        PRIVATE
        "${RVMATH_BINARY_DIR}/include"
)
