#!/usr/bin/env python3
"""LangGraph 워크플로우 시각화 스크립트

AutonomousOrchestrator의 LangGraph 워크플로우를 Mermaid 다이어그램으로 출력합니다.
"""

import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Set environment variables for minimal config
# Note: OPENROUTER_API_KEY must be set via environment variable
if not os.getenv("OPENROUTER_API_KEY"):
    raise ValueError("OPENROUTER_API_KEY environment variable is required")
os.environ.setdefault("LLM_MODEL", "google/gemini-2.5-flash-lite")
os.environ.setdefault("LLM_TEMPERATURE", "0.7")
os.environ.setdefault("LLM_MAX_TOKENS", "4000")
# Note: GOOGLE_API_KEY must be set via environment variable
if not os.getenv("GOOGLE_API_KEY"):
    raise ValueError("GOOGLE_API_KEY environment variable is required")
os.environ.setdefault("PLANNING_MODEL", "google/gemini-2.5-flash-lite")
os.environ.setdefault("REASONING_MODEL", "google/gemini-2.5-flash-lite")
os.environ.setdefault("VERIFICATION_MODEL", "google/gemini-2.5-flash-lite")
os.environ.setdefault("GENERATION_MODEL", "google/gemini-2.5-flash-lite")
os.environ.setdefault("COMPRESSION_MODEL", "google/gemini-2.5-flash-lite")
os.environ.setdefault("BUDGET_LIMIT", "10.0")
os.environ.setdefault("ENABLE_COST_OPTIMIZATION", "true")
os.environ.setdefault("MAX_RESEARCHERS", "3")
os.environ.setdefault("MIN_RESEARCHERS", "1")
os.environ.setdefault("ENABLE_ADAPTIVE_ALLOCATION", "true")
os.environ.setdefault("COMPLEXITY_THRESHOLD_HIGH", "7.0")
os.environ.setdefault("COMPLEXITY_THRESHOLD_MEDIUM", "4.0")
os.environ.setdefault("ENABLE_DYNAMIC_PRIORITY", "true")
os.environ.setdefault("MAX_PARALLEL_TASKS", "5")
os.environ.setdefault("MAX_SEARCH_DEPTH", "3")
os.environ.setdefault("MAX_SEARCH_RESULTS", "10")
os.environ.setdefault("ENABLE_DEEP_RESEARCH", "true")
os.environ.setdefault("RESEARCH_TIMEOUT", "300")
os.environ.setdefault("MCP_SERVERS", "")
os.environ.setdefault("MCP_ACADEMIC_TOOLS", "")
os.environ.setdefault("MCP_SEARCH_TOOLS", "")
os.environ.setdefault("MCP_DATA_TOOLS", "")
os.environ.setdefault("MCP_CODE_TOOLS", "")
os.environ.setdefault("MCP_BUSINESS_TOOLS", "")
os.environ.setdefault("MCP_BUILDER_ENABLED", "true")
os.environ.setdefault("MCP_BUILDER_TEMP_DIR", "temp/mcp_servers")
os.environ.setdefault("MCP_BUILDER_AUTO_CLEANUP", "true")
os.environ.setdefault("MCP_BUILDER_CACHE_ENABLED", "true")
os.environ.setdefault("ENABLE_HIERARCHICAL_COMPRESSION", "true")
os.environ.setdefault("COMPRESSION_LEVELS", "3")
os.environ.setdefault("PRESERVE_IMPORTANT_INFO", "true")
os.environ.setdefault("ENABLE_COMPRESSION_VALIDATION", "true")
os.environ.setdefault("COMPRESSION_HISTORY_ENABLED", "true")
os.environ.setdefault("MIN_COMPRESSION_RATIO", "0.3")
os.environ.setdefault("TARGET_COMPRESSION_RATIO", "0.5")
os.environ.setdefault("ENABLE_CONTINUOUS_VERIFICATION", "true")
os.environ.setdefault("VERIFICATION_STAGES", "3")
os.environ.setdefault("CONFIDENCE_THRESHOLD", "0.7")
os.environ.setdefault("ENABLE_EARLY_WARNING", "true")
os.environ.setdefault("ENABLE_FACT_CHECK", "true")
os.environ.setdefault("ENABLE_UNCERTAINTY_MARKING", "true")
os.environ.setdefault("ENABLE_ADAPTIVE_CONTEXT", "true")
os.environ.setdefault("MIN_TOKENS", "1000")
os.environ.setdefault("MAX_TOKENS", "100000")
os.environ.setdefault("IMPORTANCE_BASED_PRESERVATION", "true")
os.environ.setdefault("ENABLE_AUTO_COMPRESSION", "true")
os.environ.setdefault("ENABLE_LONG_TERM_MEMORY", "false")
os.environ.setdefault("MEMORY_REFRESH_INTERVAL", "3600")
os.environ.setdefault("ENABLE_PRODUCTION_RELIABILITY", "true")
os.environ.setdefault("ENABLE_CIRCUIT_BREAKER", "true")
os.environ.setdefault("ENABLE_EXPONENTIAL_BACKOFF", "true")
os.environ.setdefault("ENABLE_STATE_PERSISTENCE", "false")
os.environ.setdefault("ENABLE_HEALTH_CHECK", "true")
os.environ.setdefault("ENABLE_GRACEFUL_DEGRADATION", "true")
os.environ.setdefault("ENABLE_DETAILED_LOGGING", "true")
os.environ.setdefault("FAILURE_THRESHOLD", "3")
os.environ.setdefault("RECOVERY_TIMEOUT", "60")
os.environ.setdefault("STATE_BACKEND", "memory")
os.environ.setdefault("STATE_TTL", "3600")
os.environ.setdefault("OUTPUT_DIR", "./output")
os.environ.setdefault("ENABLE_PDF", "true")
os.environ.setdefault("ENABLE_MARKDOWN", "true")
os.environ.setdefault("ENABLE_JSON", "true")
os.environ.setdefault("ENABLE_DOCX", "true")
os.environ.setdefault("ENABLE_HTML", "true")
os.environ.setdefault("ENABLE_LATEX", "false")


def visualize_langgraph():
    """LangGraph 워크플로우를 시각화합니다."""
    print("=" * 80)
    print("LangGraph 워크플로우 시각화")
    print("=" * 80)

    try:
        from src.core.autonomous_orchestrator import AutonomousOrchestrator
        from src.core.researcher_config import load_config_from_env

        # Load configuration first
        print("\n[0] Configuration 로드 중...")
        config = load_config_from_env()
        print("✅ Configuration 로드 완료")

        # Create orchestrator (this builds the graph)
        print("\n[1] AutonomousOrchestrator 초기화 중...")
        orchestrator = AutonomousOrchestrator()
        orchestrator.ensure_legacy_langgraph_workflow()

        if orchestrator.graph is None:
            print("❌ 그래프가 생성되지 않았습니다.")
            return

        print("✅ 그래프 생성 완료")

        # Get graph representation
        print("\n[2] 그래프 구조 분석 중...")
        graph = orchestrator.graph.get_graph()

        # Print nodes
        print("\n" + "=" * 80)
        print("📊 노드 목록 (Nodes)")
        print("=" * 80)
        nodes = graph.nodes
        for i, node in enumerate(nodes, 1):
            print(f"{i:2d}. {node}")

        # Print edges
        print("\n" + "=" * 80)
        print("🔗 엣지 목록 (Edges)")
        print("=" * 80)
        edges = graph.edges
        for i, edge in enumerate(edges, 1):
            print(f"{i:2d}. {edge.source} → {edge.target}")

        # Generate Mermaid diagram
        print("\n" + "=" * 80)
        print("🎨 Mermaid 다이어그램")
        print("=" * 80)
        try:
            mermaid = graph.draw_mermaid()
            print(mermaid)

            # Save Mermaid file
            mermaid_file = project_root / "langgraph_workflow.mmd"
            with open(mermaid_file, "w", encoding="utf-8") as f:
                f.write(mermaid)
            print(f"\n✅ Mermaid 다이어그램 저장: {mermaid_file}")

        except Exception as e:
            print(f"⚠️ Mermaid 생성 실패: {e}")

        # Generate PNG using draw_mermaid_png
        print("\n" + "=" * 80)
        print("🖼️ PNG 이미지 생성")
        print("=" * 80)
        try:
            png_output = project_root / "langgraph_workflow.png"

            # Try using draw_mermaid_png if available
            try:
                png_data = graph.draw_mermaid_png()
                with open(png_output, "wb") as f:
                    f.write(png_data)
                print(f"✅ PNG 이미지 저장 (draw_mermaid_png): {png_output}")
            except AttributeError:
                print("⚠️ draw_mermaid_png 메서드를 사용할 수 없습니다.")
                print("대체 방법 시도 중...")

                # Alternative: use mermaid-cli if available
                import subprocess

                mermaid_file = project_root / "langgraph_workflow.mmd"

                # Check if mmdc (mermaid-cli) is installed
                try:
                    result = subprocess.run(
                        ["which", "mmdc"], capture_output=True, text=True
                    )
                    if result.returncode == 0:
                        print("✅ mermaid-cli (mmdc) 발견")
                        subprocess.run(
                            [
                                "mmdc",
                                "-i",
                                str(mermaid_file),
                                "-o",
                                str(png_output),
                                "-b",
                                "transparent",
                            ],
                            check=True,
                        )
                        print(f"✅ PNG 이미지 저장 (mmdc): {png_output}")
                    else:
                        print("⚠️ mermaid-cli가 설치되어 있지 않습니다.")
                        print("\n설치 방법:")
                        print("  npm install -g @mermaid-js/mermaid-cli")
                        print("\n또는 Python 패키지 사용:")
                        print("  pip install pyppeteer")
                        print("\n그 후 다시 실행하세요.")

                        # Try using Python mermaid package
                        try:
                            print("\nPython mermaid 패키지로 시도 중...")
                            import base64
                            from io import BytesIO

                            # Try pyppeteer-based rendering
                            print("pyppeteer 기반 렌더링 시도...")
                            subprocess.run(
                                [
                                    "python3",
                                    "-m",
                                    "pip",
                                    "install",
                                    "pyppeteer",
                                    "--quiet",
                                ],
                                check=False,
                            )

                            # Create a simple HTML file with mermaid
                            html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <script src="https://cdn.jsdelivr.net/npm/mermaid/dist/mermaid.min.js"></script>
    <script>mermaid.initialize({{startOnLoad:true}});</script>
</head>
<body>
    <div class="mermaid">
    {mermaid}
    </div>
</body>
</html>
"""
                            html_file = project_root / "langgraph_workflow.html"
                            with open(html_file, "w", encoding="utf-8") as f:
                                f.write(html_content)
                            print(f"✅ HTML 파일 저장: {html_file}")
                            print(
                                "\n📝 브라우저에서 HTML 파일을 열고 스크린샷을 찍거나,"
                            )
                            print("   온라인 도구를 사용하세요:")
                            print("   - https://mermaid.live/ (온라인 에디터)")
                            print("   - https://mermaid.ink/ (PNG 생성 API)")

                        except Exception as e3:
                            print(f"⚠️ Python mermaid 렌더링 실패: {e3}")

                except Exception as e2:
                    print(f"⚠️ 대체 방법 실패: {e2}")

        except Exception as e:
            print(f"❌ PNG 생성 실패: {e}")
            import traceback

            traceback.print_exc()

        # Print workflow summary
        print("\n" + "=" * 80)
        print("📋 워크플로우 요약")
        print("=" * 80)
        print(f"총 노드 수: {len(nodes)}")
        print(f"총 엣지 수: {len(edges)}")

        # Identify entry and end points
        entry_nodes = [n for n in nodes if not any(e.target == n for e in edges)]
        end_nodes = [n for n in nodes if not any(e.source == n for e in edges)]

        print(f"\n진입점 (Entry Points): {entry_nodes}")
        print(f"종료점 (End Points): {end_nodes}")

        # Analyze conditional edges
        print("\n조건부 라우팅:")
        for edge in edges:
            if hasattr(edge, "data") and edge.data:
                print(f"  {edge.source} --[{edge.data}]--> {edge.target}")

        print("\n" + "=" * 80)
        print("✅ 시각화 완료!")
        print("=" * 80)

    except ImportError as e:
        print(f"❌ Import 에러: {e}")
        print("\n필요한 패키지를 설치하세요:")
        print("  pip install langgraph langchain-core")
    except Exception as e:
        print(f"❌ 오류 발생: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    visualize_langgraph()
