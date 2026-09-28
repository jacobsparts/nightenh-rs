//! CUDA backend: device memory and the two embedded fatbins.
//!
//! The driver bindings, the context, the modules, the buffers and the launch
//! marshalling (`vm::{Args, Launch}`) all come from `lightgpu`; what remains here
//! is the f32-facing surface the graph walk in [`crate::exec_gpu`] expects (a
//! buffer counted in ELEMENTS rather than bytes) plus the ASCII-art of this
//! engine's own two fatbins. Compiled only with `--features cuda`;
//! `libcuda.so.1` is `dlopen`ed by lightgpu at run time, so the CPU path needs no
//! NVIDIA driver at all.

#![allow(dead_code)]

use lightgpu::ffi::CUdeviceptr;
use lightgpu::vm;

/// The toolkit fatbin produced by build.rs: the generic ops this engine uses
/// (`TOOLKIT_KERNELS`). Its `-gencode` set is sm_61 / sm_75 / sm_80 SASS plus
/// compute_80 PTX for forward compatibility.
pub static TOOLKIT_FATBIN: &[u8] =
    include_bytes!(concat!(env!("OUT_DIR"), "/nightenh_toolkit.fatbin"));

/// This engine's own kernel family (`cuda/nightenh.cu`), loaded as a SECOND
/// module. Separate modules are separate namespaces, so a name in one cannot
/// shadow a name in the other, and a name that is in neither fails eagerly at
/// startup rather than at the launch that first needs it - which is what
/// [`Cuda::init`]'s `KERNEL_NAMES` check is for. A kernel left out of build.rs's
/// `--entries` list is PRUNED from the fatbin and a name that is not there
/// resolves to nothing, so listing them here turns a mid-run failure into a
/// startup one.
pub static PROJECT_FATBIN: &[u8] =
    include_bytes!(concat!(env!("OUT_DIR"), "/nightenh_project.fatbin"));

/// Every kernel this engine launches, `(module, name)`: the toolkit's generic ops
/// and this project's own. An entry here is a kernel embedded in the binary, so
/// the two lists are also a size budget and must stay in step with `exec_gpu`'s
/// `match` - a launch of a name absent from here would still work (the module has
/// it) but a name here that the module lacks is a build/plan mismatch worth
/// catching before a picture is on screen.
pub const KERNEL_NAMES: &[(&str, &str)] = &[
    ("toolkit", "lg_relu"),
    ("toolkit", "lg_add"),
    ("toolkit", "lg_conv1x1"),
    ("toolkit", "lg_channel_mean"),
    ("toolkit", "lg_upsample2x_nearest"),
    ("toolkit", "lg_linear"),
    ("project", "ne_conv3x3_refl"),
    ("project", "ne_conv7x7_refl"),
    ("project", "ne_down3x3_refl"),
    ("project", "ne_down7x7_refl"),
    ("project", "ne_instance_norm"),
    ("project", "ne_adailn"),
    ("project", "ne_channel_affine"),
    ("project", "ne_channel_max"),
    ("project", "ne_channel_mul"),
    ("project", "ne_tanh_add"),
];

/// Device information reported at startup.
#[derive(Clone, Debug)]
pub struct DeviceInfo {
    pub index: i32,
    pub name: String,
    pub cc_major: i32,
    pub cc_minor: i32,
    pub sm_count: i32,
}

pub struct Cuda {
    /// The toolkit module (generic ops). Kernel handles are memoized in here -
    /// lightgpu caches every lookup, which matters because a forward pass
    /// resolves names on the order of a hundred times.
    pub toolkit: vm::Module,
    /// This engine's own family, from `cuda/nightenh.cu`.
    pub project: vm::Module,
    pub info: DeviceInfo,
}

impl Cuda {
    /// Bring up the driver, device 0, and both embedded fatbins.
    ///
    /// The context comes from `vm::init()`, which binds device 0's PRIMARY
    /// context. An explicit `cuCtxCreate` context would be a second, older device
    /// state and is what the toolkit deliberately avoids.
    pub fn init(verbose: bool) -> Result<Cuda, String> {
        vm::init()?;
        let dev = vm::device()?;
        let toolkit = vm::Module::load(TOOLKIT_FATBIN)?;
        let project = vm::Module::load(PROJECT_FATBIN)?;
        let info = DeviceInfo {
            index: 0,
            name: dev.name.clone(),
            cc_major: dev.cc_major,
            cc_minor: dev.cc_minor,
            sm_count: dev.sm_count,
        };
        if verbose {
            eprintln!(
                "cuda: {} cc {}.{} ({} SMs, fatbins {} + {} bytes)",
                info.name,
                info.cc_major,
                info.cc_minor,
                info.sm_count,
                TOOLKIT_FATBIN.len(),
                PROJECT_FATBIN.len()
            );
        }
        Ok(Cuda { toolkit, project, info })
    }

    /// Resolve and CHECK every kernel this engine can launch.
    ///
    /// Called once at construction so a pruned or misnamed kernel is a startup
    /// error naming the kernel, rather than a launch failure three layers into a
    /// forward pass. `Module::kernel` is lightgpu's memoized lookup.
    pub fn check_kernels(&self) -> Result<(), String> {
        for &(module, name) in KERNEL_NAMES {
            let m = if module == "toolkit" { &self.toolkit } else { &self.project };
            if !m.has(name) {
                return Err(format!(
                    "{name} is not in the {module} fatbin (build.rs's --entries list                      and exec_gpu's launches have to agree)"
                ));
            }
        }
        Ok(())
    }

    pub fn sync(&self) -> Result<(), String> {
        vm::sync()
    }
}

/// A device buffer of f32, counted in ELEMENTS (lightgpu counts bytes; the graph
/// code here thinks in tensor lengths, so the conversion stays in one place).
pub struct DevBuf {
    pub ptr: CUdeviceptr,
    pub len: usize,
    inner: Option<vm::DevBuf>,
}

impl DevBuf {
    /// A zero-element buffer: the graph uses one as the "nothing to add"
    /// sentinel (a weightless bias), and it must not call cuMemAlloc for it.
    pub fn empty() -> DevBuf {
        DevBuf { ptr: 0, len: 0, inner: None }
    }

    pub fn alloc(len: usize) -> Result<DevBuf, String> {
        if len == 0 {
            return Ok(DevBuf::empty());
        }
        let b = vm::DevBuf::alloc(len * std::mem::size_of::<f32>())?;
        Ok(DevBuf { ptr: b.ptr, len, inner: Some(b) })
    }

    pub fn from_host(v: &[f32]) -> Result<DevBuf, String> {
        let b = DevBuf::alloc(v.len())?;
        b.upload(v)?;
        Ok(b)
    }

    pub fn upload(&self, v: &[f32]) -> Result<(), String> {
        if v.len() != self.len {
            return Err(format!("upload size {} != buffer {}", v.len(), self.len));
        }
        match &self.inner {
            Some(b) => b.upload(v),
            None => Ok(()),
        }
    }

    pub fn download(&self, out: &mut [f32]) -> Result<(), String> {
        if out.len() != self.len {
            return Err(format!("download size {} != buffer {}", out.len(), self.len));
        }
        match &self.inner {
            Some(b) => b.download(out),
            None => Ok(()),
        }
    }
}
