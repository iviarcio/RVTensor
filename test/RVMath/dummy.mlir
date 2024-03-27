// RUN: rvmath-opt %s | rvmath-opt | FileCheck %s

module {
    // CHECK-LABEL: func @bar()
    func.func @bar() {
        %0 = arith.constant 1 : i32
        // CHECK: %{{.*}} = rvmath.foo %{{.*}} : i32
        %res = rvmath.foo %0 : i32
        return
    }

    // CHECK-LABEL: func @rvmath_types(%arg0: !rvmath.custom<"10">)
    func.func @rvmath_types(%arg0: !rvmath.custom<"10">) {
        return
    }
}
