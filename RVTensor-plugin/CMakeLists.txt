add_llvm_library(RVTensorPlugin
        # BUILDTREE_ONLY is only for testing purposes
        MODULE BUILDTREE_ONLY
        rvtensor-plugin.cpp

        DEPENDS
        MLIRRVTensor
        PLUGIN_TOOL
        mlir-opt

        LINK_LIBS
        MLIRRVTensor
        )
target_include_directories(
        RVTensorPlugin
        PRIVATE
        "${RVTENSOR_BINARY_DIR}/include"
)
