//! Thread priority hints. On macOS a thread with the "utility" QoS class is scheduled
//! preferably on the efficiency cores: slower, but several times less power and heat.
//! Threads a thread creates afterwards (e.g. onnxruntime's pool) inherit its class.

#[cfg(target_os = "macos")]
extern "C" {
    fn pthread_set_qos_class_self_np(class: u32, relative_priority: i32) -> i32;
}

/// `QOS_CLASS_UTILITY` from <sys/qos.h>.
#[cfg(target_os = "macos")]
const QOS_CLASS_UTILITY: u32 = 0x11;

/// Mark the calling thread as background work (efficiency cores on Apple Silicon).
/// Returns whether the hint was applied (always false on other platforms).
pub fn prefer_efficiency_cores() -> bool {
    #[cfg(target_os = "macos")]
    {
        // SAFETY: plain libpthread call on the current thread with a valid QoS constant.
        unsafe { pthread_set_qos_class_self_np(QOS_CLASS_UTILITY, 0) == 0 }
    }
    #[cfg(not(target_os = "macos"))]
    {
        false
    }
}
