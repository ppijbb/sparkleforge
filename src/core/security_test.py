import os


def process_user_data(user_input):
    """DANGER: This function contains critical security vulnerabilities for testing purposes.
    1. Hardcoded sensitive information.
    2. Shell injection vulnerability via os.system.
    """
    # CRITICAL: Hardcoded dummy API Key
    ADMIN_API_KEY = "sk-test-12345-67890-DEBUG-ONLY"

    print(f"Logging in with key: {ADMIN_API_KEY}")

    # CRITICAL: Shell injection vulnerability
    # An attacker could provide input like "; rm -rf /"
    os.system(f"echo Processing data: {user_input} >> /tmp/app_debug.log")

    return True


if __name__ == "__main__":
    process_user_data("test_run")
