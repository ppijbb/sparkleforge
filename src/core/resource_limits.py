"""
Input size limits for OOM protection

Prevents memory exhaustion when users pipe large files or import massive content.
A 10MB limit provides reasonable headroom for most use cases while preventing
catastrophic memory usage.

Inspired by mdflow's limits.ts pattern.
"""


# Maximum sizes in bytes
MAX_INPUT_SIZE = 10 * 1024 * 1024  # 10MB
MAX_OUTPUT_SIZE = 50 * 1024 * 1024  # 50MB
MAX_CODE_SIZE = 1 * 1024 * 1024  # 1MB

# Human-readable sizes for error messages
MAX_INPUT_SIZE_HUMAN = "10MB"
MAX_OUTPUT_SIZE_HUMAN = "50MB"
MAX_CODE_SIZE_HUMAN = "1MB"


class StdinSizeLimitError(Exception):
    """Error thrown when stdin input exceeds the size limit"""
    
    def __init__(self, bytes_read: int):
        self.bytes_read = bytes_read
        message = (
            f"Input exceeds {MAX_INPUT_SIZE_HUMAN} limit "
            f"(read {format_bytes(bytes_read)} so far). "
            f"Use a file path argument instead of piping large content."
        )
        super().__init__(message)
        self.name = "StdinSizeLimitError"


class FileSizeLimitError(Exception):
    """Error thrown when a file import exceeds the size limit"""
    
    def __init__(self, file_path: str, file_size: int):
        self.file_path = file_path
        self.file_size = file_size
        message = (
            f'File "{file_path}" exceeds {MAX_INPUT_SIZE_HUMAN} limit '
            f"({format_bytes(file_size)}). "
            f"Consider using line ranges or symbol extraction "
            f"to import only the relevant portion."
        )
        super().__init__(message)
        self.name = "FileSizeLimitError"


class CodeSizeLimitError(Exception):
    """Error thrown when code size exceeds the limit"""
    
    def __init__(self, code_size: int):
        self.code_size = code_size
        message = (
            f"Code size ({format_bytes(code_size)}) exceeds limit "
            f"({MAX_CODE_SIZE_HUMAN}). "
            f"Please reduce the code size or split into smaller chunks."
        )
        super().__init__(message)
        self.name = "CodeSizeLimitError"


def format_bytes(bytes_count: int) -> str:
    """
    Format bytes as human-readable string
    
    Args:
        bytes_count: Number of bytes
        
    Returns:
        Human-readable string (e.g., "1.5MB", "512KB", "1024 bytes")
    """
    if bytes_count < 1024:
        return f"{bytes_count} bytes"
    if bytes_count < 1024 * 1024:
        return f"{(bytes_count / 1024):.1f}KB"
    return f"{(bytes_count / (1024 * 1024)):.1f}MB"


def exceeds_input_limit(bytes_count: int) -> bool:
    """
    Check if a size exceeds the input limit
    
    Args:
        bytes_count: Number of bytes to check
        
    Returns:
        True if exceeds limit
    """
    return bytes_count > MAX_INPUT_SIZE


def exceeds_output_limit(bytes_count: int) -> bool:
    """
    Check if a size exceeds the output limit
    
    Args:
        bytes_count: Number of bytes to check
        
    Returns:
        True if exceeds limit
    """
    return bytes_count > MAX_OUTPUT_SIZE


def exceeds_code_limit(bytes_count: int) -> bool:
    """
    Check if a code size exceeds the limit
    
    Args:
        bytes_count: Number of bytes to check
        
    Returns:
        True if exceeds limit
    """
    return bytes_count > MAX_CODE_SIZE


class ResourceLimits:
    """
    Resource limits checker and formatter
    
    Usage:
        if ResourceLimits.exceeds_code_limit(len(code.encode())):
            raise CodeSizeLimitError(len(code.encode()))
    """
    
    MAX_INPUT_SIZE = MAX_INPUT_SIZE
    MAX_OUTPUT_SIZE = MAX_OUTPUT_SIZE
    MAX_CODE_SIZE = MAX_CODE_SIZE
    
    MAX_INPUT_SIZE_HUMAN = MAX_INPUT_SIZE_HUMAN
    MAX_OUTPUT_SIZE_HUMAN = MAX_OUTPUT_SIZE_HUMAN
    MAX_CODE_SIZE_HUMAN = MAX_CODE_SIZE_HUMAN
    
    @staticmethod
    def format_bytes(bytes_count: int) -> str:
        """Format bytes as human-readable string"""
        return format_bytes(bytes_count)
    
    @staticmethod
    def exceeds_input_limit(bytes_count: int) -> bool:
        """Check if input size exceeds limit"""
        return exceeds_input_limit(bytes_count)
    
    @staticmethod
    def exceeds_output_limit(bytes_count: int) -> bool:
        """Check if output size exceeds limit"""
        return exceeds_output_limit(bytes_count)
    
    @staticmethod
    def exceeds_code_limit(bytes_count: int) -> bool:
        """Check if code size exceeds limit"""
        return exceeds_code_limit(bytes_count)
    
    @staticmethod
    def check_input_size(data: bytes) -> bool:
        """
        Check input size and raise exception if exceeds limit
        
        Args:
            data: Input data as bytes
            
        Returns:
            True if within limit
            
        Raises:
            StdinSizeLimitError: If exceeds limit
        """
        if exceeds_input_limit(len(data)):
            raise StdinSizeLimitError(len(data))
        return True
    
    @staticmethod
    def check_code_size(code: str) -> bool:
        """
        Check code size and raise exception if exceeds limit
        
        Args:
            code: Code as string
            
        Returns:
            True if within limit
            
        Raises:
            CodeSizeLimitError: If exceeds limit
        """
        code_bytes = code.encode('utf-8')
        if exceeds_code_limit(len(code_bytes)):
            raise CodeSizeLimitError(len(code_bytes))
        return True

