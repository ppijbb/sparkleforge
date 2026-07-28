"""Forge Master Token Minimizer - Token Optimization & Context Compression Engine

외부 AI 도구/CLI 에이전트 호출 시 토큰 소모를 최소화하기 위한 프롬프트 압축,
Diff 전용 문맥 슬라이싱, 출력 결과 증류(Distillation) 컴포넌트
"""

import re
from typing import Any, Dict, List


class TokenMinimizer:
    """토큰 최소화 및 문맥 압축 관리자"""

    def __init__(self, target_reduction_ratio: float = 0.5):
        """
        Args:
            target_reduction_ratio: 목표 토큰 절감 비율 (기본값 50%)
        """
        self.target_reduction_ratio = target_reduction_ratio

    def compact_prompt(self, prompt: str, max_chars: int = 1500) -> str:
        """프롬프트 불필요 수식어 및 보일러플레이트 제거

        Args:
            prompt: 원본 프롬프트
            max_chars: 최대 문자 수

        Returns:
            압축된 프롬프트
        """
        if not prompt:
            return ""

        # 연속 불필요 공백/줄바꿈 정리
        cleaned = re.sub(r"\n{3,}", "\n\n", prompt)
        cleaned = re.sub(r"[ \t]+", " ", cleaned)

        # 불필요 안내문 패턴 축소
        patterns_to_strip = [
            r"please ensure that you provide a comprehensive and highly detailed response\.",
            r"make sure to explain every single detail step-by-step\.",
            r"as an ai assistant,",
        ]
        for pattern in patterns_to_strip:
            cleaned = re.sub(pattern, "", cleaned, flags=re.IGNORECASE)

        cleaned = cleaned.strip()

        if len(cleaned) > max_chars:
            head = cleaned[: int(max_chars * 0.7)]
            tail = cleaned[-int(max_chars * 0.3) :]
            cleaned = f"{head}\n\n... [중략: 토큰 최적화] ...\n\n{tail}"

        return cleaned

    def extract_diff_context(
        self, original_code: str, modified_code: str, context_lines: int = 3
    ) -> str:
        """전체 파일 대신 변형된 Diff 및 주변 문맥만 추출

        Args:
            original_code: 원본 코드
            modified_code: 수정된 코드
            context_lines: 변경점 주변 포함 줄 수

        Returns:
            Diff 기반 최소 문맥 텍스트
        """
        orig_lines = original_code.splitlines()
        mod_lines = modified_code.splitlines()

        if orig_lines == mod_lines:
            return "No changes detected."

        # 단순 diff 윈도우 생성
        diff_chunks: List[str] = []
        max_idx = max(len(orig_lines), len(mod_lines))

        i = 0
        while i < max_idx:
            orig_line = orig_lines[i] if i < len(orig_lines) else ""
            mod_line = mod_lines[i] if i < len(mod_lines) else ""

            if orig_line != mod_line:
                start = max(0, i - context_lines)
                end = min(max_idx, i + context_lines + 1)

                chunk = [f"--- Lines {start + 1} to {end} ---"]
                for idx in range(start, end):
                    o = orig_lines[idx] if idx < len(orig_lines) else None
                    m = mod_lines[idx] if idx < len(mod_lines) else None
                    if o == m:
                        chunk.append(f"  {o}")
                    else:
                        if o is not None:
                            chunk.append(f"- {o}")
                        if m is not None:
                            chunk.append(f"+ {m}")

                diff_chunks.append("\n".join(chunk))
                i = end
            else:
                i += 1

        return "\n\n".join(diff_chunks)

    def distill_response(self, raw_output: str, max_summary_bytes: int = 2000) -> str:
        """외부 CLI 에이전트의 터미널 출력을 요약/증류하여 메모리 흡수용 텍스트 반환

        Args:
            raw_output: CLI 원본 터미널 출력
            max_summary_bytes: 최대 저장 바이트

        Returns:
            핵심 요약 텍스트
        """
        if not raw_output:
            return ""

        # 터미널 ANSI 이스케이프 코드 제거
        clean_text = re.sub(r"\x1b\[[0-9;]*[mGKH]", "", raw_output)

        lines = [line.strip() for line in clean_text.splitlines() if line.strip()]

        # 오류, 결과, 리팩토링 핵심 줄 필터링
        important_keywords = ["error", "fail", "success", "modified", "created", "passed", "return"]
        key_lines = [
            line for line in lines if any(kw in line.lower() for kw in important_keywords)
        ]

        if not key_lines:
            key_lines = lines[:20] + lines[-20:] if len(lines) > 40 else lines

        distilled = "\n".join(key_lines)
        if len(distilled) > max_summary_bytes:
            distilled = distilled[:max_summary_bytes] + "\n... [Output Distilled]"

        return distilled

    def estimate_token_reduction(self, original_text: str, compressed_text: str) -> Dict[str, Any]:
        """토큰 절감량 및 절감률 추정

        Args:
            original_text: 원본 텍스트
            compressed_text: 압축 텍스트

        Returns:
            토큰 절감 메트릭
        """
        orig_words = len(original_text.split())
        comp_words = len(compressed_text.split())

        saved_words = max(0, orig_words - comp_words)
        reduction_rate = (saved_words / orig_words) * 100 if orig_words > 0 else 0.0

        return {
            "original_word_count": orig_words,
            "compressed_word_count": comp_words,
            "saved_word_count": saved_words,
            "reduction_percentage": round(reduction_rate, 2),
        }
