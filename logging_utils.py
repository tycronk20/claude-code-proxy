import logging
import sys

class Colors:
    CYAN = "\033[96m"
    BLUE = "\033[94m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    MAGENTA = "\033[95m"
    RESET = "\033[0m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"
    DIM = "\033[2m"

class MessageFilter(logging.Filter):
    def filter(self, record):
        blocked_phrases = [
            "LiteLLM completion()",
            "HTTP Request:",
            "selected model name for cost calculation",
            "utils.py",
            "cost_calculator"
        ]

        if hasattr(record, 'msg') and isinstance(record.msg, str):
            for phrase in blocked_phrases:
                if phrase in record.msg:
                    return False
        return True

class ColorizedFormatter(logging.Formatter):
    """Custom formatter to highlight model mappings"""
    BLUE = Colors.BLUE
    GREEN = Colors.GREEN
    YELLOW = Colors.YELLOW
    RED = Colors.RED
    RESET = Colors.RESET
    BOLD = Colors.BOLD

    def format(self, record):
        if record.levelno == logging.DEBUG and "MODEL MAPPING" in record.msg: # Changed from logging.debug to logging.DEBUG
            return f"{self.BOLD}{self.GREEN}{record.msg}{self.RESET}"

        # Default formatting for other messages
        log_format = '%(asctime)s - %(levelname)s - %(message)s'
        if hasattr(record, 'color') and record.color: # Apply color if specified in record
            log_format = f"{record.color}{log_format}{self.RESET}"

        formatter = logging.Formatter(log_format)
        return formatter.format(record)

def setup_logging(level=logging.WARN):
    """Configures global logging settings."""
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler(sys.stdout)] # Ensure logs go to stdout
    )

    # Quieten Uvicorn loggers
    logging.getLogger("uvicorn").setLevel(logging.WARNING)
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
    logging.getLogger("uvicorn.error").setLevel(logging.WARNING)

    # Add MessageFilter to the root logger
    root_logger = logging.getLogger()
    message_filter_found = any(isinstance(f, MessageFilter) for f in root_logger.filters)
    if not message_filter_found:
        root_logger.addFilter(MessageFilter())

    # Apply ColorizedFormatter to existing handlers (typically the StreamHandler set by basicConfig)
    # Check if handlers exist and are StreamHandlers
    if root_logger.hasHandlers():
        for handler in root_logger.handlers:
            if isinstance(handler, logging.StreamHandler) and not isinstance(handler.formatter, ColorizedFormatter):
                # Only replace if it's not already a ColorizedFormatter
                handler.setFormatter(ColorizedFormatter('%(asctime)s - %(levelname)s - %(message)s'))
    else: # If no handlers, add a new one (e.g. if basicConfig didn't add one)
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(ColorizedFormatter('%(asctime)s - %(levelname)s - %(message)s'))
        root_logger.addHandler(console_handler)

    # Ensure logger for this module is also set to the correct level
    logging.getLogger(__name__).setLevel(level)
