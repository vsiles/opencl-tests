use ocl::ProQue;

#[repr(C)]
#[derive(Debug, Copy, Clone, Default, PartialEq)]
struct Color {
    r: f32,
    g: f32,
    b: f32,
}

unsafe impl ocl::OclPrm for Color {}

fn main() -> ocl::Result<()> {
    let src = r#"
    struct color {
        float r;
        float g;
        float b;
    };

    __kernel void entry_point(__global int *C, struct color c, __global float *out) {
        // Get the indexes of the current work items
        int i = get_global_id(0);
        int j = get_global_id(1);

        // Store a 1 at the relevant location
        C[i + 8 * j] = (i << 8 | j);

        if (i == 0 && j == 0) {
            *out = c.r + c.g + c.b;
        }
    }"#;

    let pro_que = ProQue::builder().src(src).dims([8, 8]).build()?;

    println!("pro_que dims: {:?}", pro_que.dims());
    println!("pro_que device: {:?}", pro_que.device());

    let buffer = pro_que.create_buffer::<i32>()?;

    let out: ocl::Buffer<f32> = pro_que.buffer_builder().len(1).build()?;

    let color = Color {
        r: 0.1,
        g: 1.0,
        b: 10.0,
    };

    let kernel = pro_que
        .kernel_builder("entry_point")
        .arg(&buffer)
        .arg(&color)
        .arg(&out)
        .build()?;

    unsafe {
        kernel.enq()?;
    }

    let mut vec = vec![0i32; buffer.len()];
    buffer.read(&mut vec).enq()?;

    let mut out2: Vec<f32> = vec![0.0];
    out.read(&mut out2).enq()?;

    println!("vec len = {}", vec.len());
    println!("out2 len = {}", out2.len());
    println!("out2: {:?}", out2);

    for i in 0..8 {
        for j in 0..8 {
            print!("0x{:X} ", vec[i + 8 * j])
        }
        println!()
    }

    Ok(())
}
