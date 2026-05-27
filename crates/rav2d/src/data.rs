use std::sync::Arc;

#[derive(Debug, Clone, Default)]
pub struct DataProps {
    pub timestamp: i64,
    pub duration: i64,
    pub offset: i64,
    pub size: usize,
    pub user_data: Option<Arc<UserData>>,
}

pub struct UserData {
    data: *const u8,
    free_callback: Box<dyn Fn(*const u8) + Send + Sync>,
}

unsafe impl Send for UserData {}
unsafe impl Sync for UserData {}

impl UserData {
    pub fn new(data: *const u8, free_callback: Box<dyn Fn(*const u8) + Send + Sync>) -> Self {
        Self {
            data,
            free_callback,
        }
    }

    pub fn data(&self) -> *const u8 {
        self.data
    }
}

impl Drop for UserData {
    fn drop(&mut self) {
        (self.free_callback)(self.data);
    }
}

impl std::fmt::Debug for UserData {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("UserData")
            .field("data", &self.data)
            .finish()
    }
}

impl DataProps {
    pub fn new() -> Self {
        Self {
            timestamp: i64::MIN,
            offset: -1,
            ..Default::default()
        }
    }

    pub fn copy_from(&mut self, src: &DataProps) {
        self.timestamp = src.timestamp;
        self.duration = src.duration;
        self.offset = src.offset;
        self.size = src.size;
        self.user_data = src.user_data.clone();
    }
}

#[derive(Clone)]
pub struct Data {
    buf: Option<Arc<Vec<u8>>>,
    offset: usize,
    len: usize,
    pub props: DataProps,
}

impl Data {
    pub fn new() -> Self {
        Self {
            buf: None,
            offset: 0,
            len: 0,
            props: DataProps::new(),
        }
    }

    pub fn create(size: usize) -> Option<Self> {
        if size > usize::MAX / 2 {
            return None;
        }
        let buf = vec![0u8; size];
        Some(Self {
            buf: Some(Arc::new(buf)),
            offset: 0,
            len: size,
            props: DataProps {
                size,
                ..DataProps::new()
            },
        })
    }

    pub fn wrap(data: Vec<u8>) -> Self {
        let len = data.len();
        Self {
            buf: Some(Arc::new(data)),
            offset: 0,
            len,
            props: DataProps {
                size: len,
                ..DataProps::new()
            },
        }
    }

    pub fn data(&self) -> Option<&[u8]> {
        self.buf
            .as_ref()
            .map(|b| &b[self.offset..self.offset + self.len])
    }

    pub fn data_mut(&mut self) -> Option<&mut [u8]> {
        let offset = self.offset;
        let len = self.len;
        self.buf.as_mut().and_then(|b| {
            Arc::get_mut(b).map(|v| &mut v[offset..offset + len])
        })
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0 || self.buf.is_none()
    }

    pub fn has_data(&self) -> bool {
        self.buf.is_some()
    }

    pub fn consume(&mut self, n: usize) {
        assert!(n <= self.len);
        self.offset += n;
        self.len -= n;
        if self.len == 0 {
            self.unref();
        }
    }

    pub fn unref(&mut self) {
        self.buf = None;
        self.offset = 0;
        self.len = 0;
        self.props = DataProps::new();
    }
}

impl Default for Data {
    fn default() -> Self {
        Self::new()
    }
}

impl std::fmt::Debug for Data {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Data")
            .field("len", &self.len)
            .field("has_data", &self.buf.is_some())
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_data_create() {
        let d = Data::create(1024).unwrap();
        assert_eq!(d.len(), 1024);
        assert!(d.has_data());
        let slice = d.data().unwrap();
        assert_eq!(slice.len(), 1024);
        assert!(slice.iter().all(|&b| b == 0));
    }

    #[test]
    fn test_data_create_too_large() {
        assert!(Data::create(usize::MAX).is_none());
    }

    #[test]
    fn test_data_wrap() {
        let d = Data::wrap(vec![1, 2, 3, 4]);
        assert_eq!(d.len(), 4);
        assert_eq!(d.data().unwrap(), &[1, 2, 3, 4]);
    }

    #[test]
    fn test_data_consume() {
        let mut d = Data::wrap(vec![10, 20, 30, 40, 50]);
        d.consume(2);
        assert_eq!(d.len(), 3);
        assert_eq!(d.data().unwrap(), &[30, 40, 50]);
    }

    #[test]
    fn test_data_consume_all() {
        let mut d = Data::wrap(vec![1, 2, 3]);
        d.consume(3);
        assert!(d.is_empty());
        assert!(!d.has_data());
    }

    #[test]
    fn test_data_unref() {
        let mut d = Data::wrap(vec![1, 2, 3]);
        d.unref();
        assert!(d.is_empty());
        assert!(!d.has_data());
    }

    #[test]
    fn test_data_clone_shared() {
        let d1 = Data::wrap(vec![1, 2, 3]);
        let d2 = d1.clone();
        assert_eq!(d1.data(), d2.data());
    }

    #[test]
    fn test_data_props_defaults() {
        let p = DataProps::new();
        assert_eq!(p.timestamp, i64::MIN);
        assert_eq!(p.offset, -1);
        assert_eq!(p.size, 0);
    }

    #[test]
    fn test_data_props_copy() {
        let src = DataProps {
            timestamp: 42,
            duration: 100,
            offset: 7,
            size: 256,
            user_data: None,
        };
        let mut dst = DataProps::new();
        dst.copy_from(&src);
        assert_eq!(dst.timestamp, 42);
        assert_eq!(dst.duration, 100);
        assert_eq!(dst.offset, 7);
        assert_eq!(dst.size, 256);
    }

    #[test]
    fn test_data_new_empty() {
        let d = Data::new();
        assert!(d.is_empty());
        assert!(!d.has_data());
        assert_eq!(d.len(), 0);
    }
}
