use std::mem;
use blazesym::symbolize;

pub const MAX_STACK_DEPTH: usize = 128;
pub const TASK_COMM_LEN: usize = 16;
const ADDR_WIDTH: usize = 16;

// A Rust version of stacktrace_event in profile.h
#[repr(C)]
pub struct StacktraceEvent {
    pub pid: u32,
    pub cpu_id: u32,
    pub timestamp: u64,
    pub comm: [u8; TASK_COMM_LEN],
    pub kstack_size: i32,
    pub ustack_size: i32,
    pub kstack: [u64; MAX_STACK_DEPTH],
    pub ustack: [u64; MAX_STACK_DEPTH],
}

pub fn handle_event(symbolizer: &symbolize::Symbolizer, data: &[u8]) -> ::std::os::raw::c_int {
    if data.len() != mem::size_of::<StacktraceEvent>() {
        eprintln!(
            "Invalid size {} != {}",
            data.len(),
            mem::size_of::<StacktraceEvent>()
        );
        return 1;
    }

    let event = unsafe { &*(data.as_ptr() as *const StacktraceEvent) };

    if event.kstack_size <= 0 && event.ustack_size <= 0 {
        return 1;
    }

    let comm = std::str::from_utf8(&event.comm).unwrap_or("<unknown>");
    let timestamp_sec = event.timestamp / 1_000_000_000;
    let timestamp_nsec = event.timestamp % 1_000_000_000;
    println!("[{}.{:09}] COMM: {} (pid={}) @ CPU {}", 
             timestamp_sec, timestamp_nsec, comm, event.pid, event.cpu_id);

    if event.kstack_size > 0 {
        println!("Kernel:");
        show_stack_trace(
            &event.kstack[0..(event.kstack_size as usize / mem::size_of::<u64>())],
            symbolizer,
            0,
        );
    } else {
        println!("No Kernel Stack");
    }

    if event.ustack_size > 0 {
        println!("Userspace:");
        show_stack_trace(
            &event.ustack[0..(event.ustack_size as usize / mem::size_of::<u64>())],
            symbolizer,
            event.pid,
        );
    } else {
        println!("No Userspace Stack");
    }

    println!();
    0
}

fn print_frame(
    name: &str,
    addr_info: Option<(blazesym::Addr, blazesym::Addr, usize)>,
    code_info: &Option<symbolize::CodeInfo>,
) {
    let code_info = code_info.as_ref().map(|code_info| {
        let path = code_info.to_path();
        let path = path.display();

        match (code_info.line, code_info.column) {
            (Some(line), Some(col)) => format!(" {path}:{line}:{col}"),
            (Some(line), None) => format!(" {path}:{line}"),
            (None, _) => format!(" {path}"),
        }
    });

    if let Some((input_addr, addr, offset)) = addr_info {
        // If we have various address information bits we have a new symbol.
        println!(
            "{input_addr:#0width$x}: {name} @ {addr:#x}+{offset:#x}{code_info}",
            code_info = code_info.as_deref().unwrap_or(""),
            width = ADDR_WIDTH
        )
    } else {
        // Otherwise we are dealing with an inlined call.
        println!(
            "{:width$}  {name}{code_info} [inlined]",
            " ",
            code_info = code_info
                .map(|info| format!(" @{info}"))
                .as_deref()
                .unwrap_or(""),
            width = ADDR_WIDTH
        )
    }
}

// Pid 0 means a kernel space stack.
fn show_stack_trace(stack: &[u64], symbolizer: &symbolize::Symbolizer, pid: u32) {
    let converted_stack;
    // The kernel always reports `u64` addresses, whereas blazesym uses `Addr`.
    // Convert the stack trace as necessary.
    let stack = if mem::size_of::<blazesym::Addr>() != mem::size_of::<u64>() {
        converted_stack = stack
            .iter()
            .copied()
            .map(|addr| addr as blazesym::Addr)
            .collect::<Vec<_>>();
        converted_stack.as_slice()
    } else {
        // SAFETY: `Addr` has the same size as `u64`, so it can be trivially and
        //         safely converted.
        unsafe { mem::transmute::<_, &[blazesym::Addr]>(stack) }
    };

    let src = if pid == 0 {
        symbolize::source::Source::from(symbolize::source::Kernel::default())
    } else {
        symbolize::source::Source::from(symbolize::source::Process::new(pid.into()))
    };

    let syms = match symbolizer.symbolize(&src, symbolize::Input::AbsAddr(stack)) {
        Ok(syms) => syms,
        Err(err) => {
            eprintln!("  failed to symbolize addresses: {err:#}");
            return;
        }
    };

    for (input_addr, sym) in stack.iter().copied().zip(syms) {
        match sym {
            symbolize::Symbolized::Sym(symbolize::Sym {
                name,
                addr,
                offset,
                code_info,
                inlined,
                ..
            }) => {
                print_frame(&name, Some((input_addr, addr, offset)), &code_info);
                for frame in inlined.iter() {
                    print_frame(&frame.name, None, &frame.code_info);
                }
            }
            symbolize::Symbolized::Unknown(..) => {
                println!("{input_addr:#0width$x}: <no-symbol>", width = ADDR_WIDTH)
            }
        }
    }
}