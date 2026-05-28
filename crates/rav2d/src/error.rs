use std::fmt;

#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum Rav2dError {
    Eof,
    Again,
    InvalidData,
    FrameTooLarge,
    InvalidParam,
    OutOfMemory,
}

impl fmt::Display for Rav2dError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Eof => write!(f, "end of stream"),
            Self::Again => write!(f, "need more data"),
            Self::InvalidData => write!(f, "invalid or corrupt bitstream data"),
            Self::FrameTooLarge => write!(f, "frame dimensions exceed limit"),
            Self::InvalidParam => write!(f, "invalid parameter"),
            Self::OutOfMemory => write!(f, "out of memory"),
        }
    }
}

impl std::error::Error for Rav2dError {}

pub type Result<T> = std::result::Result<T, Rav2dError>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_display() {
        assert_eq!(Rav2dError::Eof.to_string(), "end of stream");
        assert_eq!(Rav2dError::Again.to_string(), "need more data");
        assert_eq!(Rav2dError::InvalidData.to_string(), "invalid or corrupt bitstream data");
        assert_eq!(Rav2dError::FrameTooLarge.to_string(), "frame dimensions exceed limit");
        assert_eq!(Rav2dError::InvalidParam.to_string(), "invalid parameter");
        assert_eq!(Rav2dError::OutOfMemory.to_string(), "out of memory");
    }

    #[test]
    fn test_error_is_error_trait() {
        let err: Box<dyn std::error::Error> = Box::new(Rav2dError::InvalidData);
        assert_eq!(err.to_string(), "invalid or corrupt bitstream data");
    }

    #[test]
    fn test_error_clone_eq() {
        let e1 = Rav2dError::Eof;
        let e2 = e1.clone();
        assert_eq!(e1, e2);
        assert_ne!(e1, Rav2dError::Again);
    }
}
