import os
import subprocess
from pathlib import Path

def setup_fixture(workspace: Path) -> None:
    project_dir = workspace / "project"
    project_dir.mkdir(parents=True, exist_ok=True)
    
    # Create simple python project structure for build_env scenario
    (project_dir / "main.py").write_text(
        "import os\n"
        "from dotenv import load_dotenv\n\n"
        "load_dotenv()\n\n"
        "def main():\n"
        "    env = os.getenv('APP_ENV', 'unknown')\n"
        "    print(f'Running with APP_ENV={env}')\n"
        "    if env != 'development':\n"
        "        raise ValueError(f'Expected APP_ENV=development, got {env}')\n"
        "    print('Build environment setup successfully!')\n\n"
        "if __name__ == '__main__':\n"
        "    main()\n"
    )
    
    (project_dir / "requirements.txt").write_text(
        "python-dotenv==1.0.0\n"
    )

def verify_fixture(workspace: Path) -> dict:
    project_dir = workspace / "project"
    env_file = project_dir / ".env"
    
    env_setup = 0.0
    if env_file.exists():
        content = env_file.read_text()
        if "APP_ENV=development" in content:
            env_setup = 1.0
            
    verify_runs = 0.0
    if env_file.exists():
        try:
            res = subprocess.run(["python3", "main.py"], cwd=project_dir, capture_output=True, text=True, timeout=10)
            if res.returncode == 0 and "Build environment setup successfully!" in res.stdout:
                verify_runs = 1.0
        except Exception:
            pass
            
    return {
        "env_setup": env_setup,
        "verify_runs": verify_runs,
    }
