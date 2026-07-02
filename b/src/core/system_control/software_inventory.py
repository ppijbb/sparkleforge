import subprocess

class SoftwareInventory:
    def get_installed_packages(self):
        # Placeholder for cross-platform package inventory
        try:
            # Example: list pip packages
            result = subprocess.run(['pip', 'list', '--format=json'], capture_output=True, text=True)
            return result.stdout
        except Exception:
            return "[]"
