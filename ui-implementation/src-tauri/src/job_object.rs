//! Windows Job Object wrapper — v0.3.3 Fix A.
//!
//! Wraps the Python sidecar child in a Job Object configured with
//! `JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE`. When the parent Tauri process
//! exits for any reason — clean shutdown, force-kill, crash, BSOD
//! recovery — the OS automatically closes the job handle, and the kernel
//! terminates every process inside the job before reclaiming it.
//!
//! This kills two birds:
//!   1. Dev mode: the bundled Python sidecar holds `.pyd` file locks that
//!      block Cargo's incremental rebuild on every Python edit (os
//!      error 32). With the job, the sidecar dies the instant Tauri
//!      unloads, so the rebuild grabs uncontended locks.
//!   2. Production: an ungracefully-exited Tauri (crash, force-quit)
//!      previously left an orphan sidecar bound to port 8766. The next
//!      Tauri launch then failed because the port was held. The job
//!      makes orphans impossible.
//!
//! Non-Windows targets get no-op stubs so `main.rs` can refer to the
//! type unconditionally and `cargo build` stays green cross-platform.

#[cfg(target_os = "windows")]
mod imp {
    use std::io;
    use std::mem;
    use std::ptr;

    use windows_sys::Win32::Foundation::{CloseHandle, FALSE, HANDLE};
    use windows_sys::Win32::System::JobObjects::{
        AssignProcessToJobObject, CreateJobObjectW, SetInformationJobObject,
        JobObjectExtendedLimitInformation, JOBOBJECT_EXTENDED_LIMIT_INFORMATION,
        JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE,
    };
    use windows_sys::Win32::System::Threading::{
        OpenProcess, PROCESS_SET_QUOTA, PROCESS_TERMINATE,
    };

    /// RAII guard owning a Windows Job Object handle. Dropping the guard
    /// closes the handle, which (because of `KILL_ON_JOB_CLOSE`) causes
    /// the kernel to terminate every process currently assigned to the
    /// job. Holding this in a long-lived Tauri-managed state ties the
    /// sidecar's lifetime to the Tauri process's lifetime.
    pub struct JobObjectGuard {
        handle: HANDLE,
    }

    // SAFETY: HANDLE is `*mut c_void`. The only mutations through the
    // pointer are in `create()` (returning Self) and `drop()` (closing
    // the handle). All other operations (`assign`) are read-only on the
    // guard. Win32 guarantees concurrent calls to `AssignProcessToJobObject`
    // and `CloseHandle` on the same job handle are safe, and the guard
    // is held behind a Mutex at the call site anyway.
    unsafe impl Send for JobObjectGuard {}
    unsafe impl Sync for JobObjectGuard {}

    impl JobObjectGuard {
        /// Create an unnamed Job Object with `JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE`
        /// pre-set so any subsequently-assigned process is bound to this
        /// guard's lifetime.
        pub fn create() -> io::Result<Self> {
            unsafe {
                let handle = CreateJobObjectW(ptr::null(), ptr::null());
                if handle == 0 {
                    return Err(io::Error::last_os_error());
                }

                let mut info: JOBOBJECT_EXTENDED_LIMIT_INFORMATION = mem::zeroed();
                info.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE;

                let ok = SetInformationJobObject(
                    handle,
                    JobObjectExtendedLimitInformation,
                    &info as *const _ as *const _,
                    mem::size_of::<JOBOBJECT_EXTENDED_LIMIT_INFORMATION>() as u32,
                );
                if ok == 0 {
                    let err = io::Error::last_os_error();
                    CloseHandle(handle);
                    return Err(err);
                }

                Ok(Self { handle })
            }
        }

        /// Add a child process (by PID) to this job. The process and any
        /// of its descendants will be killed when this guard drops.
        ///
        /// Requires the child to NOT have been spawned with
        /// `CREATE_BREAKAWAY_FROM_JOB`. Roampal's spawn path uses default
        /// flags so this is fine.
        pub fn assign(&self, pid: u32) -> io::Result<()> {
            unsafe {
                let proc_handle = OpenProcess(
                    PROCESS_SET_QUOTA | PROCESS_TERMINATE,
                    FALSE,
                    pid,
                );
                if proc_handle == 0 {
                    return Err(io::Error::last_os_error());
                }

                let ok = AssignProcessToJobObject(self.handle, proc_handle);
                let err = if ok == 0 { Some(io::Error::last_os_error()) } else { None };

                // Close our local reference to the process handle; the job
                // retains its own internal reference, so the process stays
                // tracked until the job itself is closed.
                CloseHandle(proc_handle);

                if let Some(e) = err {
                    return Err(e);
                }
                Ok(())
            }
        }
    }

    impl Drop for JobObjectGuard {
        fn drop(&mut self) {
            // Closing the job handle while KILL_ON_JOB_CLOSE is set asks
            // the kernel to terminate every process in the job before
            // the handle is reclaimed.
            unsafe {
                CloseHandle(self.handle);
            }
        }
    }
}

#[cfg(target_os = "windows")]
pub use imp::JobObjectGuard;

// Non-Windows stub — Roampal's `start_backend` is already gated to Windows
// only, but keeping a uniform type lets `main.rs` reference it without
// peppering `#[cfg]` everywhere.
#[cfg(not(target_os = "windows"))]
pub struct JobObjectGuard;

#[cfg(not(target_os = "windows"))]
impl JobObjectGuard {
    pub fn create() -> std::io::Result<Self> {
        Ok(Self)
    }

    pub fn assign(&self, _pid: u32) -> std::io::Result<()> {
        Ok(())
    }
}
