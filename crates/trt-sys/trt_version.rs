// Pure text parser for NvInferVersion.h, split out of build.rs so it can be
// unit-tested (cargo never builds a build script as a test harness): build.rs
// include!s it, and src/lib.rs pulls it into a #[cfg(test)] module by path.
use std::collections::HashMap;

/// Parse "MAJOR.MINOR.PATCH.BUILD" from NvInferVersion.h text. TRT >= 10.13
/// defines the NV_TENSORRT_* macros through one level of indirection
/// (`#define NV_TENSORRT_MAJOR TRT_MAJOR_ENTERPRISE`), so values that aren't
/// numeric literals are resolved through the define map.
#[allow(dead_code)] // used by build.rs via include!; lib.rs only sees it under cfg(test)
pub(crate) fn parse_trt_version_text(text: &str) -> Option<String> {
    let defines: HashMap<&str, &str> = text
        .lines()
        .filter_map(|line| {
            let mut words = line.split_whitespace();
            if words.next()? != "#define" {
                return None;
            }
            Some((words.next()?, words.next()?))
        })
        .collect();
    let grab = |name: &str| -> Option<u32> {
        let mut value = *defines.get(name)?;
        for _ in 0..16 {
            if let Ok(number) = value.parse() {
                return Some(number);
            }
            value = *defines.get(value)?;
        }
        None
    };
    Some(format!(
        "{}.{}.{}.{}",
        grab("NV_TENSORRT_MAJOR")?,
        grab("NV_TENSORRT_MINOR")?,
        grab("NV_TENSORRT_PATCH")?,
        grab("NV_TENSORRT_BUILD")?,
    ))
}
