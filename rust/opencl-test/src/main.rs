use ocl::ProQue;

fn main() -> ocl::Result<()> {
    let src = r#"
    __kernel void entry_point(__global int *C) {
        // Get the indexes of the current work items
        int i = get_global_id(0);
        int j = get_global_id(1);

        // Store a 1 at the relevant location
        C[i + 8 * j] = (i << 8 | j);
    }"#;

    let pro_que = ProQue::builder().src(src).dims([8, 8]).build()?;

    println!("pro_que dims: {:?}", pro_que.dims());
    println!("pro_que device: {:?}", pro_que.device());

    let buffer = pro_que.create_buffer::<i32>()?;

    let kernel = pro_que.kernel_builder("entry_point").arg(&buffer).build()?;

    unsafe {
        kernel.enq()?;
    }

    let mut vec = vec![0i32; buffer.len()];
    buffer.read(&mut vec).enq()?;

    for i in 0..8 {
        for j in 0..8 {
            print!("0x{:X} ", vec[i + 8 * j])
        }
        println!()
    }

    Ok(())
}
