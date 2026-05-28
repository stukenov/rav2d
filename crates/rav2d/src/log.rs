use std::fmt;

pub type LogCallback = Box<dyn Fn(&str) + Send + Sync>;

pub struct Logger {
    callback: Option<LogCallback>,
}

impl Logger {
    pub fn new() -> Self {
        Self { callback: None }
    }

    pub fn with_default() -> Self {
        Self {
            callback: Some(Box::new(|msg| {
                eprint!("{}", msg);
            })),
        }
    }

    pub fn with_callback(callback: LogCallback) -> Self {
        Self {
            callback: Some(callback),
        }
    }

    pub fn log(&self, args: fmt::Arguments<'_>) {
        if let Some(ref cb) = self.callback {
            cb(&fmt::format(args));
        }
    }
}

impl Default for Logger {
    fn default() -> Self {
        Self::with_default()
    }
}

impl fmt::Debug for Logger {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Logger")
            .field("has_callback", &self.callback.is_some())
            .finish()
    }
}

#[allow(unused_macros)]
macro_rules! rav2d_log {
    ($logger:expr, $($arg:tt)*) => {
        $logger.log(format_args!($($arg)*))
    };
}

#[allow(unused_imports)]
pub(crate) use rav2d_log;

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::{Arc, Mutex};

    #[test]
    fn test_default_logger_has_callback() {
        let logger = Logger::with_default();
        assert!(logger.callback.is_some());
    }

    #[test]
    fn test_new_logger_no_callback() {
        let logger = Logger::new();
        assert!(logger.callback.is_none());
    }

    #[test]
    fn test_log_no_callback_noop() {
        let logger = Logger::new();
        rav2d_log!(logger, "should not panic {}", 42);
    }

    #[test]
    fn test_log_with_callback() {
        let captured = Arc::new(Mutex::new(String::new()));
        let captured_clone = captured.clone();
        let logger = Logger::with_callback(Box::new(move |msg| {
            captured_clone.lock().unwrap().push_str(msg);
        }));
        rav2d_log!(logger, "hello {}", 42);
        assert_eq!(*captured.lock().unwrap(), "hello 42");
    }
}
