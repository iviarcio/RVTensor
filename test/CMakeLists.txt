llvm_canonicalize_cmake_booleans(
        MLIR_ENABLE_BINDINGS_PYTHON
)

configure_lit_site_cfg(
        ${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.py.in
        ${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg.py
        MAIN_CONFIG
        ${CMAKE_CURRENT_SOURCE_DIR}/lit.cfg.py
)

set(RVMATH_TEST_DEPENDS
        FileCheck count not
        rvmath-capi-test
        rvmath-opt
        rvmath-translate
        RVMathPlugin
        )
if(MLIR_ENABLE_BINDINGS_PYTHON)
  list(APPEND RVMATH_TEST_DEPENDS RVMathPythonModules)
endif()

add_lit_testsuite(check-rvmath "Running the rvmath regression tests"
        ${CMAKE_CURRENT_BINARY_DIR}
        DEPENDS ${RVMATH_TEST_DEPENDS}
        )
set_target_properties(check-rvmath PROPERTIES FOLDER "Tests")

add_lit_testsuites(RVMATH ${CMAKE_CURRENT_SOURCE_DIR} DEPENDS ${RVMATH_TEST_DEPENDS})

add_subdirectory(CAPI)
