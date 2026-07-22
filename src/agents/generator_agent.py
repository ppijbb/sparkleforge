"""Generator Agent (Synthesizer).

수집된 모든 정보와 검증된 결과를 종합하여 최종 리포트 및 결과물을 생성합니다.
"""

import logging
import logging
import os
import re
import subprocess
from pathlib import Path
from typing import Any, Dict

from src.core.harness_state import HarnessState

logger = logging.getLogger(__name__)


class GeneratorAgent:
    """최종 결과물 생성 및 종합 에이전트"""

    def __init__(self):
        self.name = "generator_agent"

    async def synthesize(self, state: HarnessState) -> Dict[str, Any]:
        logger.info(
            f"[{self.name}] 📝 Synthesizing final results and extracting generated artifacts..."
        )

        # 동적 출력 디렉토리 설정
        current_output_dir = Path(state["meta"].get("output_dir", "output/default"))
        current_output_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir = current_output_dir

        # 실제로는 state의 context나 results에서 내용을 읽어와 LLM 프롬프트 생성
        final_output = state["workflow"].get(
            "final_output", "Final compiled report based on tasks."
        )
        tasks = state["workflow"].get("tasks", [])

        outputs = []
        for i, task in enumerate(tasks):
            result = task.get("result")
            status = task.get("status")
            task.get("description", "")
            if status == "completed" and result:
                # If there's python code in the result, try saving it.
                if isinstance(result, dict) and result.get("content"):
                    outputs.append(result.get("content"))
                elif isinstance(result, str):
                    outputs.append(result)

        # 합친 내용
        combined_result = "\n".join(outputs)

        # 모든 파이썬 코드 블록 추출
        python_blocks = re.findall(
            r"```python\s+(.*?)\s+```", combined_result, re.DOTALL | re.IGNORECASE
        )

        if python_blocks:
            logger.info(f"[{self.name}] Found {len(python_blocks)} python code blocks.")

            # PPT 생성 코드가 있는지 확인
            ppt_block = None
            for block in python_blocks:
                if "Presentation()" in block or "pptx" in block:
                    ppt_block = block
                    break

            # 가급적 PPT 블록을 prioritieze
            code_to_run = ppt_block if ppt_block else python_blocks[0]
            script_name = "ppt_generator_script.py" if ppt_block else "extracted_code.py"
            script_path = self.output_dir / script_name

            try:

                def fix_path(match):
                    path = match.group(1)
                    # 파일명만 있거나 이미 'output/'로 시작하는 경우 처리
                    filename = os.path.basename(path)
                    return f"{match.group(0).split('(')[0]}('{str(self.output_dir)}/{filename}')"

                # .save('path') 및 .savefig('path') 패턴 처리
                code_to_run = re.sub(r"\.save\(['\"]([^'\"]+)['\"]\)", fix_path, code_to_run)
                code_to_run = re.sub(r"\.savefig\(['\"]([^'\"]+)['\"]\)", fix_path, code_to_run)

                # add_picture('path', ...) 처리 (추가 인자 대응)
                def fix_add_picture(match):
                    path = match.group(1)
                    filename = os.path.basename(path)
                    new_path = f"{str(self.output_dir)}/{filename}"
                    return match.group(0).replace(match.group(1), new_path)

                code_to_run = re.sub(
                    r"add_picture\(['\"]([^'\"]+)['\"]", fix_add_picture, code_to_run
                )

                with open(script_path, "w", encoding="utf-8") as f:
                    f.write(code_to_run)
                logger.info(f"[{self.name}] Successfully wrote {script_path}!")

                # 가상환경의 python 경로 사용
                venv_python = os.path.join(os.getcwd(), ".venv", "bin", "python3")
                if not os.path.exists(venv_python):
                    venv_python = sys.executable or "python3"

                logger.info(f"[{self.name}] Executing {script_path} using {venv_python}...")
                result = subprocess.run(
                    [venv_python, str(script_path)],
                    cwd=str(self.output_dir),
                    capture_output=True,
                    text=True,
                    check=False,
                )
                if result.returncode != 0:
                    logger.error(
                        "[%s] Generator script failed (exit %d): %s",
                        self.name,
                        result.returncode,
                        result.stderr,
                    )
                    raise RuntimeError(
                        f"Script execution failed with exit code {result.returncode}"
                    )

                final_output += f"\n\n[Auto-Generated Artifacts]: Saved and executed codebase in {self.output_dir}/"
            except Exception as e:
                logger.error(f"Failed to extract and run code: {e}")

        # Marp Markdown 생성 (혁신 보너스)
        marp_content = self.generate_marp(combined_result)
        marp_path = self.output_dir / "presentation_marp.md"
        with open(marp_path, "w", encoding="utf-8") as f:
            f.write(marp_content)
        logger.info(f"[{self.name}] Marp presentation saved to {marp_path}")
        final_output += f"\n- Marp presentation: {marp_path}"

        return {
            "final_output": final_output,
            "synthesized_content": combined_result,
            "marp_path": str(marp_path),
        }

    def generate_marp(self, text: str) -> str:
        """Create a Marp-compatible markdown presentation."""
        header = "---\nmarp: true\ntheme: uncover\nclass: invert\npaginate: true\n---\n\n"

        # PPT와 중복되는 코드 블록은 Marp에서 불필요할 수 있으므로 제거하거나 정리
        text_clean = re.sub(r"```python.*?```", "", text, flags=re.DOTALL)

        # 섹션별 슬라이드 분리
        sections = re.split(r"(^#+\s+.*)", text_clean, flags=re.MULTILINE)

        slides = []
        current_slide = ""
        for part in sections:
            if not part.strip():
                continue
            if part.startswith("#"):
                if current_slide:
                    slides.append(current_slide)
                current_slide = part
            else:
                current_slide += "\n" + part

        if current_slide:
            slides.append(current_slide)

        marp_body = "\n\n---\n\n".join(slides)
        return header + marp_body
