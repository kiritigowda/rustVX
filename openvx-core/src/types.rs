//! OpenVX Types

/// OpenVX Status codes
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
pub enum VxStatus {
    #[error("Success")]
    Success = 0,
    #[error("Invalid reference")]
    ErrorInvalidReference = -1,
    #[error("Invalid parameters")]
    ErrorInvalidParameters = -2,
    #[error("Invalid graph")]
    ErrorInvalidGraph = -3,
    #[error("Invalid node")]
    ErrorInvalidNode = -4,
    #[error("Invalid kernel")]
    ErrorInvalidKernel = -5,
    #[error("Invalid context")]
    ErrorInvalidContext = -6,
    #[error("Invalid format")]
    ErrorInvalidFormat = -7,
    #[error("Invalid dimension")]
    ErrorInvalidDimension = -8,
    #[error("Invalid kernel parameters")]
    ErrorInvalidKernelParameters = -9,
    #[error("Not implemented")]
    ErrorNotImplemented = -30,
    #[error("Out of memory")]
    ErrorNoMemory = -36,
}

impl VxStatus {
    pub fn is_success(&self) -> bool {
        matches!(self, VxStatus::Success)
    }
    pub fn is_error(&self) -> bool {
        !self.is_success()
    }
}

pub type VxResult<T> = Result<T, VxStatus>;

/// OpenVX Type identifiers
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u32)]
pub enum VxType {
    Invalid = 0,
    Context = 1,
    Graph = 2,
    Node = 3,
    Kernel = 4,
    Parameter = 5,
    Delay = 6,
    Lut = 7,
    Distribution = 8,
    Image = 9,
    Buffer = 10,
    Pyramid = 11,
    Threshold = 12,
    Matrix = 13,
    Convolution = 14,
    Scalar = 15,
    Array = 16,
    ObjectArray = 17,
    Tensor = 18,
    Reference = 19,
    MetaFormat = 20,
}

/// OpenVX Kernel enumerations
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u32)]
pub enum VxKernel {
    ColorConvert = 0,
    ChannelExtract = 1,
    ChannelCombine = 2,
    Sobel3x3 = 3,
    Magnitude = 4,
    Phase = 5,
    ScaleImage = 6,
    Add = 7,
    Subtract = 8,
    Multiply = 9,
    WeightedAverage = 10,
    Convolve = 11,
    Gaussian3x3 = 12,
    Median3x3 = 13,
    Dilate3x3 = 14,
    Erode3x3 = 15,
    Histogram = 16,
    EqualizeHistogram = 17,
    IntegralImage = 18,
    MeanStdDev = 19,
    MinMaxLoc = 20,
    AbsDiff = 21,
    MeanShift = 22,
    Threshold = 23,
    IntegralImageSq = 24,
    Box3x3 = 25,
    Gaussian5x5 = 26,
    Sobel5x5 = 27,
    Laplacian = 28,
    NonLinearFilter = 29,
    WarpAffine = 30,
    WarpPerspective = 31,
    HarrisCorners = 32,
    FASTCorners = 33,
    OpticalFlowPyrLK = 34,
    Remap = 35,
    CornerMinEigenVal = 36,
    HoughLinesP = 37,
    CannyEdgeDetector = 38,
    And = 39,
    Or = 40,
    Xor = 41,
    Not = 42,
    // Enhanced Vision (OpenVX 1.2+)
    Min = 43,
    Max = 44,
}

/// Reference type
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VxReferenceType {
    Image = 0,
    Graph = 1,
    Node = 2,
    Context = 3,
    Kernel = 4,
    Parameter = 5,
    Array = 6,
    Scalar = 7,
    Threshold = 8,
    Matrix = 9,
    Pyramid = 10,
    Remap = 11,
    Delay = 12,
    Tensor = 13,
    ObjectArray = 14,
    Distribution = 15,
}

/// Border modes
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VxBorderMode {
    Undefined = 0,
    Constant = 1,
    Replicate = 2,
}

/// Color space types
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VxColorSpace {
    RGB = 0,
    YUV = 1,
    YCbCr = 2,
    HSV = 3,
    HSL = 4,
    Lab = 5,
}

/// Image format
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VxImageFormat {
    RGB = 0,
    RGBX = 1,
    RGBA = 2,
    YUV4 = 3,
    NV12 = 4,
    NV21 = 5,
    UYVY = 6,
    YUYV = 7,
    IYUV = 8,
    Grayscale = 9,
}

impl VxImageFormat {
    pub fn channels(&self) -> usize {
        match self {
            VxImageFormat::Grayscale => 1,
            VxImageFormat::RGB => 3,
            VxImageFormat::RGBX | VxImageFormat::RGBA => 4,
            VxImageFormat::YUV4 | VxImageFormat::NV12 | VxImageFormat::NV21 => 3,
            VxImageFormat::UYVY | VxImageFormat::YUYV => 2,
            VxImageFormat::IYUV => 3,
        }
    }

    pub fn bytes_per_pixel(&self) -> usize {
        match self {
            VxImageFormat::Grayscale => 1,
            VxImageFormat::RGB => 3,
            VxImageFormat::RGBX | VxImageFormat::RGBA => 4,
            VxImageFormat::YUV4 => 3,
            VxImageFormat::NV12 | VxImageFormat::NV21 => 1, // Luma only
            VxImageFormat::UYVY | VxImageFormat::YUYV => 2,
            VxImageFormat::IYUV => 1, // Luma only
        }
    }
}

/// Error types
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VxError {
    InvalidReference = -1,
    InvalidParameters = -2,
    InvalidGraph = -3,
    InvalidNode = -4,
    InvalidKernel = -5,
    InvalidContext = -6,
    InvalidCoordinates = -7,
    NotAllocated = -8,
    NotAvailable = -9,
    NotImplemented = -100,
    InvalidValue = -11,
    InvalidDimension = -12,
    InvalidFormat = -13,
    NoMemory = -14,
}

impl std::fmt::Display for VxError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            VxError::InvalidReference => write!(f, "Invalid reference"),
            VxError::InvalidParameters => write!(f, "Invalid parameters"),
            VxError::InvalidGraph => write!(f, "Invalid graph"),
            VxError::InvalidNode => write!(f, "Invalid node"),
            VxError::InvalidKernel => write!(f, "Invalid kernel"),
            VxError::InvalidContext => write!(f, "Invalid context"),
            VxError::InvalidCoordinates => write!(f, "Invalid coordinates"),
            VxError::NotAllocated => write!(f, "Not allocated"),
            VxError::NotAvailable => write!(f, "Not available"),
            VxError::NotImplemented => write!(f, "Not implemented"),
            VxError::InvalidValue => write!(f, "Invalid value"),
            VxError::InvalidDimension => write!(f, "Invalid dimension"),
            VxError::InvalidFormat => write!(f, "Invalid format"),
            VxError::NoMemory => write!(f, "Out of memory"),
        }
    }
}

impl std::error::Error for VxError {}
