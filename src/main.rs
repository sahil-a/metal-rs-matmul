use half::f16;
use metal::*;
use objc::rc::autoreleasepool;
use std::mem;
use std::path::Path;
use std::path::PathBuf;
use std::time::Instant;

/// A reusable context holding Metal device, command queue, and precompiled pipelines.
pub struct MetalContext {
    device: Device,
    command_queue: CommandQueue,
    dot_product_pipeline: ComputePipelineState,
    matrix_multiply_pipeline: ComputePipelineState,
}

impl MetalContext {
    /// Create a new context from a `.metallib` file.
    pub fn new<P: AsRef<Path>>(library_path: P) -> Self {
        autoreleasepool(|| {
            // 1. Get the default system GPU device
            let device = Device::system_default().expect("No Metal-capable device found!");

            // 2. Create a command queue
            let command_queue = device.new_command_queue();

            // 3. Load the `.metallib` file
            let library_path = PathBuf::from(library_path.as_ref());
            let library = device
                .new_library_with_file(library_path)
                .expect("Failed to load metallib");

            // 4. Create pipeline for dot_product kernel
            let dot_kernel = library.get_function("dot_product", None).unwrap();
            let dot_product_pipeline = device
                .new_compute_pipeline_state_with_function(&dot_kernel)
                .unwrap();

            // 5. Create pipeline for matrix_multiply kernel
            let mat_kernel = library.get_function("matrix_multiply", None).unwrap();
            let matrix_multiply_pipeline = device
                .new_compute_pipeline_state_with_function(&mat_kernel)
                .unwrap();

            Self {
                device,
                command_queue,
                dot_product_pipeline,
                matrix_multiply_pipeline,
            }
        })
    }

    /// Compute the dot product of two half-precision vectors on the GPU.
    ///
    /// Returns the resulting sum as a `u32`.
    pub fn dot_product(&self, a: &[f16], b: &[f16]) -> u32 {
        assert_eq!(a.len(), b.len(), "Vectors must have the same length!");
        let array_len = a.len() as u32;

        autoreleasepool(|| {
            // 1. Create buffers for input/output
            let input_buffer_a = create_buffer(&self.device, a);
            let input_buffer_b = create_buffer(&self.device, b);
            let output_buffer = create_buffer(&self.device, &[0u32]);
            let arraylen_buffer = create_buffer(&self.device, &[array_len]);

            // 2. Create command buffer & encoder
            let command_buffer = self.command_queue.new_command_buffer();
            let encoder = command_buffer.new_compute_command_encoder();

            // 3. Set pipeline & buffers
            encoder.set_compute_pipeline_state(&self.dot_product_pipeline);
            encoder.set_buffer(0, Some(&input_buffer_a), 0);
            encoder.set_buffer(1, Some(&input_buffer_b), 0);
            encoder.set_buffer(2, Some(&output_buffer), 0);
            encoder.set_buffer(3, Some(&arraylen_buffer), 0);

            // 4. Determine thread layout
            let num_threads = self.dot_product_pipeline.thread_execution_width();
            let threadgroup_size = MTLSize {
                width: num_threads,
                height: 1,
                depth: 1,
            };
            // Round up to cover all elements
            let threadgroup_count = MTLSize {
                width: ((array_len as u64 + num_threads - 1) / num_threads) as u64,
                height: 1,
                depth: 1,
            };

            // 5. Allocate threadgroup memory (for reduction)
            encoder.set_threadgroup_memory_length(
                0,
                threadgroup_size.width * (mem::size_of::<f16>() as u64),
            );

            // 6. Encode & dispatch
            encoder.dispatch_thread_groups(threadgroup_count, threadgroup_size);
            encoder.end_encoding();

            // 7. Commit & wait
            command_buffer.commit();
            command_buffer.wait_until_completed();

            // 8. Read result
            let ptr = output_buffer.contents() as *mut u32;
            unsafe { *ptr }
        })
    }

    /// Multiply two matrices (A of size row_len x inner_len, and B of size inner_len x col_len)
    /// stored in row-major order. Both inputs are `Vec<f16>`; output is `Vec<u32>` (the sum can fit in 32 bits).
    ///
    /// Returns a `row_len * col_len` vector of `u32`.
    pub fn matrix_multiply(
        &self,
        a: &[f16],
        b: &[f16],
        row_len: u32,
        inner_len: u32,
        col_len: u32,
    ) -> Vec<f16> {
        // Sanity checks
        assert_eq!(
            a.len() as u32,
            row_len * inner_len,
            "Dimensions of A are incorrect."
        );
        assert_eq!(
            b.len() as u32,
            inner_len * col_len,
            "Dimensions of B are incorrect."
        );

        let out_len = row_len * col_len;

        autoreleasepool(|| {
            // 1. Create buffers
            let input_buffer_a = create_buffer(&self.device, a);
            let input_buffer_b = create_buffer(&self.device, b);
            let output_buffer = create_buffer(
                &self.device,
                vec![f16::from_f32(0.0); out_len as usize].as_slice(),
            );
            let row_len_buffer = create_buffer(&self.device, &[row_len]);
            let inner_len_buffer = create_buffer(&self.device, &[inner_len]);
            let col_len_buffer = create_buffer(&self.device, &[col_len]);

            // 2. Create command buffer & encoder
            let command_buffer = self.command_queue.new_command_buffer();
            let encoder = command_buffer.new_compute_command_encoder();

            // 3. Set pipeline & buffers
            encoder.set_compute_pipeline_state(&self.matrix_multiply_pipeline);
            encoder.set_buffer(0, Some(&input_buffer_a), 0);
            encoder.set_buffer(1, Some(&input_buffer_b), 0);
            encoder.set_buffer(2, Some(&output_buffer), 0);
            encoder.set_buffer(3, Some(&row_len_buffer), 0);
            encoder.set_buffer(4, Some(&inner_len_buffer), 0);
            encoder.set_buffer(5, Some(&col_len_buffer), 0);

            // 4. Determine thread layout
            //    We'll dispatch (row_len x col_len) threadgroups, each having 'inner_len' threads.
            let threadgroup_count = MTLSize {
                width: row_len as u64,
                height: col_len as u64,
                depth: 1,
            };
            let threadgroup_size = MTLSize {
                width: inner_len as u64,
                height: 1,
                depth: 1,
            };

            // 5. Allocate threadgroup memory
            //    We allocate 'inner_len' worth of half-precision data for partial sums.
            encoder.set_threadgroup_memory_length(
                0,
                threadgroup_size.width * (mem::size_of::<f16>() as u64),
            );

            // 6. Encode & dispatch
            encoder.dispatch_thread_groups(threadgroup_count, threadgroup_size);
            encoder.end_encoding();

            // 7. Commit & wait
            command_buffer.commit();
            command_buffer.wait_until_completed();

            // 8. Read results
            let ptr = output_buffer.contents() as *const f16;
            let output_slice = unsafe { std::slice::from_raw_parts(ptr, out_len as usize) };
            output_slice.to_vec()
        })
    }
}

/// A helper to create a Metal buffer from a slice of data.
/// Uses the correct byte size for `T`.
fn create_buffer<T: Copy>(device: &Device, data: &[T]) -> Buffer {
    let size = (data.len() * std::mem::size_of::<T>()) as u64;
    let raw_ptr = data.as_ptr() as *const std::ffi::c_void;
    device.new_buffer_with_data(raw_ptr, size, MTLResourceOptions::CPUCacheModeDefaultCache)
}

fn main() {
    // Create the shared Metal context
    let context = MetalContext::new("sumshader.metallib");

    // Configure some matrix sizes that you want to test
    let row_len = 512;
    let inner_len = 512;
    let col_len = 512;

    // Create some test data
    let mat_a = vec![f16::from_f32(2.0); (row_len * inner_len) as usize];
    let mat_b = vec![f16::from_f32(4.0); (inner_len * col_len) as usize];

    // Number of times to repeat for the benchmark
    let iterations = 10;

    // Optional warmup to ensure GPU is “warmed up”
    for _ in 0..2 {
        let _ = context.matrix_multiply(&mat_a, &mat_b, row_len, inner_len, col_len);
    }

    // Now perform the actual benchmark
    let start = Instant::now();
    for _ in 0..iterations {
        let _ = context.matrix_multiply(&mat_a, &mat_b, row_len, inner_len, col_len);
    }
    let total_time = start.elapsed();
    let avg_time = total_time / iterations;

    // Compute a rough FLOP count.
    // Each output element requires `inner_len` multiply-add pairs => 2 * inner_len ops per output element.
    // For a row_len x col_len result, total_ops = 2 * row_len * col_len * inner_len.
    let total_ops = 2.0 * row_len as f64 * col_len as f64 * inner_len as f64;

    // Convert average time to seconds
    let avg_time_s = avg_time.as_secs_f64();

    // GFLOPS = (total_ops / time_in_seconds) / 1e9
    // but note that we do this on a "per dispatch" basis using avg_time,
    // so it's for a single multiplication's worth of ops.
    let gflops = (total_ops / avg_time_s) / 1e9;

    println!(
        "Ran {iterations} matrix multiplies of size {row_len}x{inner_len} * {inner_len}x{col_len} \
         in {:#?} total; ~{:#?} each => approx. {:.2} GFLOPS",
        total_time, avg_time, gflops
    );
}
