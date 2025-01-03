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


// this naive implementation has the following limitations:
// 1. inner len limitation - 1024 threads per group
// 2. no tiling
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

// todo: vectorization
kernel void matrix_multiply_tiled(device half *a [[ buffer(0) ]], 
                            device half *b [[ buffer(1) ]],
                            device half *output [[ buffer(2) ]],
                            device uint *row_len [[ buffer(3) ]],
                            device uint *inner_len [[ buffer(4) ]],
                            device uint *col_len [[ buffer(5) ]],
                            device uint *tile_size [[ buffer(6) ]],
                            uint2 tid [[ threadgroup_position_in_grid ]],
                            uint2 lid [[ thread_position_in_threadgroup ]],
                            uint2 threads_per_threadgroup [[ threads_per_threadgroup ]],
                            // the below needs to be set to (tile_size**2) length in rs
                            threadgroup half *shared_mem_a [[ threadgroup(0) ]],
                            // the below needs to be set to (tile_size**2) length in rs
                            threadgroup half *shared_mem_b [[ threadgroup(1) ]],
                            // the below needs to be set to threads length in rs
                            threadgroup half *shared_products [[ threadgroup(2) ]])
{

    uint items_per_thread = (*tile_size + threads_per_threadgroup.x - 1) / threads_per_threadgroup.x;
    for (uint inner_start = 0; inner_start < *inner_len; inner_start += *tile_size) {
        // 1. Load all rows of A and cols of B for this tile
        // within shared memory, rows of A are first, then columns of B
        for (uint tile = 0; tile < *tile_size; tile++) {
            uint x = tid.x * *tile_size + tile;
            uint y = tid.y * *tile_size + tile;

            if (x < *row_len) { // load row x
                for (uint i = 0; i < items_per_thread; i++) {
                    uint idx = items_per_thread * lid.x + i;
                    if (idx+inner_start < *inner_len) {
                        // load element of row x of A into shared mem
                        shared_mem_a[*tile_size * tile + idx] = a[x * *inner_len + idx + inner_start];
                    }
                }
            }
            if (y < *col_len) { // load col y
                for (uint i = 0; i < items_per_thread; i++) {
                    uint idx = items_per_thread * lid.x + i;
                    if (idx+inner_start < *inner_len) {
                        // load element of col y of B into shared mem
                        shared_mem_b[*tile_size * tile + idx] = b[(idx+inner_start) * *col_len + y];
                    }
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);


        // 2. Compute output for each tile serially
        for (uint tile_x = 0; tile_x < *tile_size; tile_x++) {
            for (uint tile_y = 0; tile_y < *tile_size; tile_y++) {
                uint x = tid.x * *tile_size + tile_x;
                uint y = tid.y * *tile_size + tile_y;

                shared_products[lid.x] = half(0.0); // zero out previous results
                if (x < *row_len && y < *col_len) {
                    // store products in shared products
                    for (uint i = 0; i < items_per_thread; i++) {
                        uint idx = items_per_thread * lid.x + i;
                        if (idx+inner_start < *inner_len) {
                            shared_products[lid.x] += shared_mem_a[*tile_size * tile_x + idx] * shared_mem_b[*tile_size * tile_y + idx];
                        }
                    }
                    threadgroup_barrier(mem_flags::mem_threadgroup); // this is within a uniform condition

                    // parallel reduction within the thread group to shared_products[0]
                    for (uint stride = 2; stride/2 < threads_per_threadgroup.x; stride <<= 1) {
                        if (lid.x % stride == 0 && (lid.x + stride/2 < threads_per_threadgroup.x)) {
                            shared_products[lid.x] += shared_products[lid.x + stride/2];
                        }
                        threadgroup_barrier(mem_flags::mem_threadgroup);
                    }

                    // write the final result to the output
                    if (lid.x == 0) {
                        output[x * *col_len + y] += shared_products[0];
                    }
                }
            }
        }
    }
}
