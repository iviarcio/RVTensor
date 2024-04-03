llvm_canonicalize_cmake_booleans(
        MLIR_ENABLE_BINDINGS_PYTHON
)

configure_lit_site_cfg(
        ${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.py.in
        ${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg.py
        MAIN_CONFIG
        ${CMAKE_CURRENT_SOURCE_DIR}/lit.cfg.py
)

set(RVTENSOR_TEST_DEPENDS
        FileCheck count not
        rvtensor-capi-test
        rvtensor-opt
        rvtensor-translate
        RVTensorPlugin
        )
if(MLIR_ENABLE_BINDINGS_PYTHON)
  list(APPEND RVTENSOR_TEST_DEPENDS RVTensorPythonModules)
endif()

add_lit_testsuite(check-rvtensor "Running the rvtensor regression tests"
        ${CMAKE_CURRENT_BINARY_DIR}
        DEPENDS ${RVTENSOR_TEST_DEPENDS}
        )
set_target_properties(check-rvtensor PROPERTIES FOLDER "Tests")

add_lit_testsuites(RVTENSOR ${CMAKE_CURRENT_SOURCE_DIR} DEPENDS ${RVTENSOR_TEST_DEPENDS})

add_subdirectory(CAPI)
