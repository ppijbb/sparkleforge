import os


def process_user_data(user_input):
    """Process user data safely.

    This function previously contained critical security vulnerabilities
    (hardcoded credentials and shell injection via os.system) used as a
    fixture for the automated zero-day exploit surface scanner. It has
    been hardened so the forged microservice no longer exposes an
    exploitable surface before deployment.
    """
    import logging
    import shlex
    import subprocess

    # Resolve the admin API key from the environment instead of hardcoding it.
    ADMIN_API_KEY = os.environ.get("ADMIN_API_KEY")
    if not ADMIN_API_KEY:
        raise RuntimeError("ADMIN_API_KEY environment variable is required")

    # Never log the full secret; only confirm that a key was configured.
    logging.info("Logging in with configured ADMIN_API_KEY")

    # Safe append to the debug log without invoking a shell, preventing
    # shell injection from arbitrary user input.
    log_path = os.path.join(os.environ.get("APP_DEBUG_LOG", "/tmp/app_debug.log"))
    sanitized_input = shlex.split(user_input) if isinstance(user_input, str) else [str(user_input)]
    with open(log_path, "a", encoding="utf-8") as log_file:
        log_file.write("Processing data: " + " ".join(sanitized_input) + "\n")

    return True


if __name__ == "__main__":
    process_user_data("test_run")
