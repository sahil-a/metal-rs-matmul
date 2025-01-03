#include <metal_stdlib>

using namespace metal;

// to compile: 
// xcrun metal -c sumshader.metal -o sumshader.air
// && xcrun metallib sumshader.air -o sumshader.metallib

// naive sum
kernel void sum(device half *data [[ buffer(0) ]],
                volatile device atomic_uint *sum [[ buffer(1) ]],
                uint gid [[ thread_position_in_grid ]])
{
    atomic_fetch_add_explicit(sum, (uint)data[gid], memory_order_relaxed);
}

// 2 3 5 1 6 9 2 2 9 (example which terminates at stride=32)
// 5 3 6 1 15 9 4 2 9 (stride = 2)
// 11 3 6 1 19 9 4 2 9 (stride = 4)
// 30 3 6 1 19 9 4 2 9 (stride = 8)
// 39 3 6 1 19 9 4 2 9 (stride = 16)
// note - contiguous memory access would have been better
// like adding the second half to the first half repeatedly
kernel void sum_parallel(device half *data [[ buffer(0) ]], 
                            volatile device atomic_uint *sum [[ buffer(1) ]],
                            device uint *array_len [[ buffer(2) ]],
                            uint gid [[ thread_position_in_grid ]],
                            uint tid [[ threadgroup_position_in_grid ]],
                            uint lid [[ thread_position_in_threadgroup ]],
                            uint threads_per_threadgroup [[ threads_per_threadgroup ]],
                            //uint simd_per_threadgroup [[ simdgroups_per_threadgroup ]],
                            threadgroup half *shared_mem [[ threadgroup(0) ]])
{
    // this thread group should load all data
    if (gid < *array_len) {
        shared_mem[lid] = data[gid]; // for dot product, you would multiply here
    } else {
        shared_mem[lid] = 0;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // parallel reduction within each threadgroup
    for (uint stride = 2; stride/2 < threads_per_threadgroup; stride <<= 1) {
        if (lid % stride == 0 && (lid + stride/2 < threads_per_threadgroup)) {
            shared_mem[lid] += shared_mem[lid + stride/2];
        }
        // synchronization needed per stride
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    
    // write the final result to the output
    if (lid == 0) {
        atomic_fetch_add_explicit(sum, (uint)shared_mem[0], memory_order_relaxed);
    }
}

// same as above, with one line change for multiplication
kernel void dot_product(device half *a [[ buffer(0) ]], 
                            device half *b [[ buffer(1) ]],
                            volatile device atomic_uint *output [[ buffer(2) ]],
                            device uint *array_len [[ buffer(3) ]],
                            uint gid [[ thread_position_in_grid ]],
                            uint tid [[ threadgroup_position_in_grid ]],
                            uint lid [[ thread_position_in_threadgroup ]],
                            uint threads_per_threadgroup [[ threads_per_threadgroup ]],
                            //uint simd_per_threadgroup [[ simdgroups_per_threadgroup ]],
                            threadgroup half *shared_mem [[ threadgroup(0) ]])
{
    // this thread group should load all data
    if (gid < *array_len) {
        shared_mem[lid] = a[gid] * b[gid];
    } else {
        shared_mem[lid] = 0;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // parallel reduction within each threadgroup
    for (uint stride = 2; stride/2 < threads_per_threadgroup; stride <<= 1) {
        if (lid % stride == 0 && (lid + stride/2 < threads_per_threadgroup)) {
            shared_mem[lid] += shared_mem[lid + stride/2];
        }
        // synchronization needed per stride
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    
    // add the final result to the output
    if (lid == 0) {
        atomic_fetch_add_explicit(output, (uint)shared_mem[0], memory_order_relaxed);
    }
}


// todo: tiling
// todo: vectorization
// todo: inner len limitation - 1024 threads per group
kernel void matrix_multiply(device half *a [[ buffer(0) ]], 
                            device half *b [[ buffer(1) ]],
                            device half *output [[ buffer(2) ]],
                            device uint *row_len [[ buffer(3) ]],
                            device uint *inner_len [[ buffer(4) ]],
                            device uint *col_len [[ buffer(5) ]],
                            uint2 tid [[ threadgroup_position_in_grid ]],
                            uint2 lid [[ thread_position_in_threadgroup ]],
                            uint2 threads_per_threadgroup [[ threads_per_threadgroup ]],
                            // the below needs to be set to inner_len length in rs
                            threadgroup half *shared_mem [[ threadgroup(0) ]])
{
    // check if this thread is within the inner dimension
    if (lid.x < *inner_len) {
        // For matrix A: row = tid.x, col = lid
        // For matrix B: row = lid, col = tid.y
        shared_mem[lid.x] = a[tid.x * *inner_len + lid.x] * b[lid.x * *col_len + tid.y];
    } else {
        shared_mem[lid.x] = 0;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // parallel reduction within each threadgroup
    for (uint stride = 2; stride/2 < threads_per_threadgroup.x; stride <<= 1) {
        if (lid.x % stride == 0 && (lid.x + stride/2 < threads_per_threadgroup.x)) {
                       shared_mem[lid.x] += shared_mem[lid.x + stride/2];
        }
        // synchronization needed per stride
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    
    // write the final result to the output at (tid.x, tid.y)
    if (lid.x == 0) {
        output[tid.x * *col_len + tid.y] = shared_mem[0];
    }
}
