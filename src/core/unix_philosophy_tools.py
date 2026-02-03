"""
Unix 철학 적용 - 제한된 목적 도구 모음

한 가지 일을 잘하는 도구 설계, 제한된 목적의 도구 모음 구성,
도구별 명확한 책임 분리, 도구 조합을 통한 복잡한 작업 수행,
맥락 이해력 향상을 위한 도구 메타데이터, Unix 스타일 도구 체인 구성
"""

import asyncio
import json
import logging
import time
import subprocess
import shlex
from typing import Dict, Any, List, Optional, Callable, Union, Protocol
from dataclasses import dataclass, field
from enum import Enum
from abc import ABC, abstractmethod
from pathlib import Path
import aiofiles

logger = logging.getLogger(__name__)


class ToolCategory(Enum):
    """도구 카테고리."""
    TEXT_PROCESSING = "text_processing"    # 텍스트 처리
    DATA_EXTRACTION = "data_extraction"    # 데이터 추출
    SEARCH_FILTERING = "search_filtering"  # 검색 및 필터링
    FORMAT_CONVERSION = "format_conversion" # 형식 변환
    VALIDATION_CHECKING = "validation_checking" # 검증 및 확인
    AGGREGATION_SUMMARY = "aggregation_summary"  # 집계 및 요약


class DataFormat(Enum):
    """데이터 형식."""
    TEXT = "text"
    JSON = "json"
    CSV = "csv"
    XML = "xml"
    HTML = "html"
    MARKDOWN = "markdown"
    YAML = "yaml"


class PipeProtocol(Protocol):
    """파이프 프로토콜."""
    async def process(self, input_data: Any) -> Any:
        """데이터 처리."""
        ...


@dataclass
class ToolMetadata:
    """도구 메타데이터."""
    name: str
    description: str
    category: ToolCategory
    input_formats: List[DataFormat]
    output_formats: List[DataFormat]
    tags: List[str] = field(default_factory=list)
    version: str = "1.0.0"
    author: Optional[str] = None
    dependencies: List[str] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)


@dataclass
class ProcessingResult:
    """처리 결과."""
    success: bool
    output_data: Any
    processing_time: float
    metadata: Dict[str, Any] = field(default_factory=dict)
    error_message: Optional[str] = None
    input_format: Optional[DataFormat] = None
    output_format: Optional[DataFormat] = None


class UnixTool(ABC):
    """Unix 스타일 도구 추상 클래스."""

    def __init__(self, metadata: ToolMetadata):
        """초기화."""
        self.metadata = metadata
        self.usage_count = 0
        self.success_count = 0
        self.total_processing_time = 0.0

    @abstractmethod
    async def process(self, input_data: Any, **kwargs) -> ProcessingResult:
        """데이터 처리 (한 가지 일만 수행)."""
        pass

    @property
    def success_rate(self) -> float:
        """성공률."""
        return self.success_count / self.usage_count if self.usage_count > 0 else 0.0

    @property
    def avg_processing_time(self) -> float:
        """평균 처리 시간."""
        return self.total_processing_time / self.usage_count if self.usage_count > 0 else 0.0

    def _record_execution(self, success: bool, processing_time: float):
        """실행 기록."""
        self.usage_count += 1
        if success:
            self.success_count += 1
        self.total_processing_time += processing_time


class TextExtractor(UnixTool):
    """텍스트 추출 도구 - 한 가지 일: 텍스트에서 특정 패턴 추출."""

    def __init__(self):
        metadata = ToolMetadata(
            name="text_extractor",
            description="텍스트에서 특정 패턴을 추출하는 도구",
            category=ToolCategory.DATA_EXTRACTION,
            input_formats=[DataFormat.TEXT, DataFormat.HTML, DataFormat.MARKDOWN],
            output_formats=[DataFormat.JSON],
            tags=["extraction", "pattern", "text"]
        )
        super().__init__(metadata)

    async def process(self, input_data: Any, pattern: str = r"(.*)", **kwargs) -> ProcessingResult:
        """텍스트에서 패턴 추출."""
        start_time = time.time()

        try:
            import re

            if isinstance(input_data, str):
                matches = re.findall(pattern, input_data, re.MULTILINE | re.DOTALL)
                result = {
                    "extracted_items": matches,
                    "count": len(matches),
                    "pattern": pattern
                }

                processing_time = time.time() - start_time
                self._record_execution(True, processing_time)

                return ProcessingResult(
                    success=True,
                    output_data=result,
                    processing_time=processing_time,
                    input_format=DataFormat.TEXT,
                    output_format=DataFormat.JSON,
                    metadata={"extraction_method": "regex", "pattern": pattern}
                )
            else:
                raise ValueError("Input must be string")

        except Exception as e:
            processing_time = time.time() - start_time
            self._record_execution(False, processing_time)

            return ProcessingResult(
                success=False,
                output_data=None,
                processing_time=processing_time,
                error_message=str(e)
            )


class JSONFormatter(UnixTool):
    """JSON 포맷터 - 한 가지 일: 데이터를 JSON 형식으로 변환."""

    def __init__(self):
        metadata = ToolMetadata(
            name="json_formatter",
            description="데이터를 JSON 형식으로 변환하는 도구",
            category=ToolCategory.FORMAT_CONVERSION,
            input_formats=[DataFormat.TEXT, DataFormat.CSV, DataFormat.YAML],
            output_formats=[DataFormat.JSON],
            tags=["format", "json", "conversion"]
        )
        super().__init__(metadata)

    async def process(self, input_data: Any, **kwargs) -> ProcessingResult:
        """JSON 형식으로 변환."""
        start_time = time.time()

        try:
            if isinstance(input_data, str):
                # JSON 파싱 시도
                try:
                    parsed = json.loads(input_data)
                    result = parsed
                    input_format = DataFormat.JSON
                except json.JSONDecodeError:
                    # 일반 텍스트를 JSON으로 변환
                    result = {"text": input_data}
                    input_format = DataFormat.TEXT

            elif isinstance(input_data, dict):
                result = input_data
                input_format = DataFormat.JSON

            elif isinstance(input_data, list):
                result = {"items": input_data}
                input_format = DataFormat.JSON

            else:
                result = {"value": str(input_data)}
                input_format = DataFormat.TEXT

            processing_time = time.time() - start_time
            self._record_execution(True, processing_time)

            return ProcessingResult(
                success=True,
                output_data=result,
                processing_time=processing_time,
                input_format=input_format,
                output_format=DataFormat.JSON,
                metadata={"conversion_type": "to_json"}
            )

        except Exception as e:
            processing_time = time.time() - start_time
            self._record_execution(False, processing_time)

            return ProcessingResult(
                success=False,
                output_data=None,
                processing_time=processing_time,
                error_message=str(e)
            )


class ContentFilter(UnixTool):
    """컨텐츠 필터 - 한 가지 일: 특정 조건에 따라 컨텐츠 필터링."""

    def __init__(self):
        metadata = ToolMetadata(
            name="content_filter",
            description="특정 조건에 따라 컨텐츠를 필터링하는 도구",
            category=ToolCategory.SEARCH_FILTERING,
            input_formats=[DataFormat.JSON, DataFormat.TEXT],
            output_formats=[DataFormat.JSON],
            tags=["filter", "search", "condition"]
        )
        super().__init__(metadata)

    async def process(self, input_data: Any, condition: str = "", **kwargs) -> ProcessingResult:
        """컨텐츠 필터링."""
        start_time = time.time()

        try:
            if isinstance(input_data, list):
                # 리스트 필터링
                if condition.startswith("len>") and condition[4:].isdigit():
                    min_length = int(condition[4:])
                    filtered = [item for item in input_data if len(str(item)) > min_length]
                elif condition.startswith("contains:"):
                    substring = condition[9:]
                    filtered = [item for item in input_data if substring in str(item)]
                else:
                    # 기본: 모든 항목 통과
                    filtered = input_data

                result = {
                    "filtered_items": filtered,
                    "original_count": len(input_data),
                    "filtered_count": len(filtered),
                    "condition": condition
                }

            elif isinstance(input_data, dict):
                # 딕셔너리 필터링
                if condition.startswith("key:"):
                    key = condition[4:]
                    filtered = {k: v for k, v in input_data.items() if key in k}
                else:
                    filtered = input_data

                result = {
                    "filtered_data": filtered,
                    "condition": condition
                }

            else:
                result = {"original_data": input_data, "condition": condition}

            processing_time = time.time() - start_time
            self._record_execution(True, processing_time)

            return ProcessingResult(
                success=True,
                output_data=result,
                processing_time=processing_time,
                input_format=DataFormat.JSON,
                output_format=DataFormat.JSON,
                metadata={"filter_condition": condition}
            )

        except Exception as e:
            processing_time = time.time() - start_time
            self._record_execution(False, processing_time)

            return ProcessingResult(
                success=False,
                output_data=None,
                processing_time=processing_time,
                error_message=str(e)
            )


class DataAggregator(UnixTool):
    """데이터 집계자 - 한 가지 일: 데이터를 집계하고 요약."""

    def __init__(self):
        metadata = ToolMetadata(
            name="data_aggregator",
            description="데이터를 집계하고 요약하는 도구",
            category=ToolCategory.AGGREGATION_SUMMARY,
            input_formats=[DataFormat.JSON, DataFormat.CSV],
            output_formats=[DataFormat.JSON],
            tags=["aggregate", "summary", "statistics"]
        )
        super().__init__(metadata)

    async def process(self, input_data: Any, **kwargs) -> ProcessingResult:
        """데이터 집계 및 요약."""
        start_time = time.time()

        try:
            if isinstance(input_data, list):
                # 리스트 집계
                if input_data and isinstance(input_data[0], dict):
                    # 딕셔너리 리스트 집계
                    summary = self._aggregate_dict_list(input_data)
                else:
                    # 일반 리스트 집계
                    summary = {
                        "count": len(input_data),
                        "unique_count": len(set(str(x) for x in input_data)),
                        "type_distribution": self._get_type_distribution(input_data)
                    }

            elif isinstance(input_data, dict):
                # 딕셔너리 집계
                summary = {
                    "key_count": len(input_data),
                    "value_types": {k: type(v).__name__ for k, v in input_data.items()},
                    "nested_structures": sum(1 for v in input_data.values() if isinstance(v, (dict, list)))
                }

            else:
                summary = {"value": input_data, "type": type(input_data).__name__}

            processing_time = time.time() - start_time
            self._record_execution(True, processing_time)

            return ProcessingResult(
                success=True,
                output_data={"summary": summary, "original_type": type(input_data).__name__},
                processing_time=processing_time,
                input_format=DataFormat.JSON,
                output_format=DataFormat.JSON,
                metadata={"aggregation_type": "summary"}
            )

        except Exception as e:
            processing_time = time.time() - start_time
            self._record_execution(False, processing_time)

            return ProcessingResult(
                success=False,
                output_data=None,
                processing_time=processing_time,
                error_message=str(e)
            )

    def _aggregate_dict_list(self, data: List[Dict]) -> Dict[str, Any]:
        """딕셔너리 리스트 집계."""
        if not data:
            return {}

        # 공통 키 찾기
        all_keys = set()
        for item in data:
            all_keys.update(item.keys())

        # 각 키별 통계
        stats = {}
        for key in all_keys:
            values = [item.get(key) for item in data if key in item]
            if values:
                if isinstance(values[0], (int, float)):
                    stats[key] = {
                        "count": len(values),
                        "sum": sum(values),
                        "avg": sum(values) / len(values),
                        "min": min(values),
                        "max": max(values)
                    }
                else:
                    stats[key] = {
                        "count": len(values),
                        "unique_count": len(set(str(v) for v in values)),
                        "type": type(values[0]).__name__
                    }

        return {
            "total_items": len(data),
            "common_keys": list(all_keys),
            "statistics": stats
        }

    def _get_type_distribution(self, data: List) -> Dict[str, int]:
        """타입 분포 계산."""
        type_counts = {}
        for item in data:
            type_name = type(item).__name__
            type_counts[type_name] = type_counts.get(type_name, 0) + 1
        return type_counts


class ValidatorTool(UnixTool):
    """검증 도구 - 한 가지 일: 데이터 검증."""

    def __init__(self):
        metadata = ToolMetadata(
            name="validator",
            description="데이터의 유효성을 검증하는 도구",
            category=ToolCategory.VALIDATION_CHECKING,
            input_formats=[DataFormat.JSON, DataFormat.TEXT],
            output_formats=[DataFormat.JSON],
            tags=["validation", "check", "verify"]
        )
        super().__init__(metadata)

    async def process(self, input_data: Any, **kwargs) -> ProcessingResult:
        """데이터 검증."""
        start_time = time.time()

        try:
            validation_result = {
                "is_valid": True,
                "checks": [],
                "errors": []
            }

            # JSON 유효성 검증
            if isinstance(input_data, str):
                try:
                    json.loads(input_data)
                    validation_result["checks"].append("json_format")
                except json.JSONDecodeError as e:
                    validation_result["is_valid"] = False
                    validation_result["errors"].append(f"Invalid JSON: {e}")

            # 구조 검증
            if isinstance(input_data, (dict, list)):
                validation_result["checks"].append("structure_check")
                # 추가 구조 검증 로직...

            # 내용 검증
            if isinstance(input_data, dict):
                required_keys = kwargs.get("required_keys", [])
                for key in required_keys:
                    if key not in input_data:
                        validation_result["is_valid"] = False
                        validation_result["errors"].append(f"Missing required key: {key}")

            processing_time = time.time() - start_time
            self._record_execution(validation_result["is_valid"], processing_time)

            return ProcessingResult(
                success=True,
                output_data=validation_result,
                processing_time=processing_time,
                input_format=DataFormat.JSON,
                output_format=DataFormat.JSON,
                metadata={"validation_checks": validation_result["checks"]}
            )

        except Exception as e:
            processing_time = time.time() - start_time
            self._record_execution(False, processing_time)

            return ProcessingResult(
                success=False,
                output_data=None,
                processing_time=processing_time,
                error_message=str(e)
            )


class ToolPipeline:
    """도구 파이프라인 - Unix 스타일 체인."""

    def __init__(self, name: str = "pipeline"):
        """초기화."""
        self.name = name
        self.tools: List[UnixTool] = []
        self.execution_history: List[Dict[str, Any]] = []
        self.created_at = time.time()

    def add_tool(self, tool: UnixTool):
        """도구 추가."""
        self.tools.append(tool)
        logger.info(f"Added tool to pipeline {self.name}: {tool.metadata.name}")

    async def execute(self, input_data: Any, **kwargs) -> Dict[str, Any]:
        """파이프라인 실행."""
        start_time = time.time()
        current_data = input_data
        execution_results = []

        logger.info(f"Starting pipeline execution: {self.name} with {len(self.tools)} tools")

        for i, tool in enumerate(self.tools):
            logger.debug(f"Executing tool {i+1}/{len(self.tools)}: {tool.metadata.name}")

            # 도구별 파라미터 추출
            tool_params = kwargs.get(tool.metadata.name, {})

            # 도구 실행
            result = await tool.process(current_data, **tool_params)

            execution_results.append({
                "tool_index": i,
                "tool_name": tool.metadata.name,
                "success": result.success,
                "processing_time": result.processing_time,
                "error": result.error_message
            })

            if not result.success:
                logger.error(f"Pipeline failed at tool {tool.metadata.name}: {result.error_message}")
                break

            # 다음 도구의 입력으로 결과 사용
            current_data = result.output_data

        total_time = time.time() - start_time

        pipeline_result = {
            "pipeline_name": self.name,
            "success": all(r["success"] for r in execution_results),
            "total_time": total_time,
            "tool_count": len(self.tools),
            "executed_tools": len(execution_results),
            "final_output": current_data,
            "execution_details": execution_results
        }

        self.execution_history.append(pipeline_result)

        logger.info(f"Pipeline {self.name} completed in {total_time:.2f}s, success: {pipeline_result['success']}")
        return pipeline_result

    def get_pipeline_stats(self) -> Dict[str, Any]:
        """파이프라인 통계."""
        if not self.execution_history:
            return {"executions": 0}

        successes = sum(1 for r in self.execution_history if r["success"])
        total_time = sum(r["total_time"] for r in self.execution_history)

        return {
            "executions": len(self.execution_history),
            "success_rate": successes / len(self.execution_history),
            "avg_execution_time": total_time / len(self.execution_history),
            "tools_in_pipeline": len(self.tools),
            "tool_names": [t.metadata.name for t in self.tools]
        }


class UnixPhilosophyToolbox:
    """
    Unix 철학 도구 모음.

    각 도구가 한 가지 일을 잘 수행하도록 설계.
    도구들을 조합하여 복잡한 작업을 수행.
    """

    def __init__(self):
        """초기화."""
        self.tools: Dict[str, UnixTool] = {}
        self.pipelines: Dict[str, ToolPipeline] = {}

        # 기본 도구들 등록
        self._register_default_tools()

        logger.info("UnixPhilosophyToolbox initialized with default tools")

    def _register_default_tools(self):
        """기본 도구들 등록."""
        self.register_tool(TextExtractor())
        self.register_tool(JSONFormatter())
        self.register_tool(ContentFilter())
        self.register_tool(DataAggregator())
        self.register_tool(ValidatorTool())

    def register_tool(self, tool: UnixTool):
        """도구 등록."""
        self.tools[tool.metadata.name] = tool
        logger.info(f"Registered Unix tool: {tool.metadata.name} ({tool.metadata.category.value})")

    def get_tool(self, name: str) -> Optional[UnixTool]:
        """도구 가져오기."""
        return self.tools.get(name)

    def create_pipeline(self, name: str, tool_names: List[str]) -> ToolPipeline:
        """파이프라인 생성."""
        pipeline = ToolPipeline(name)

        for tool_name in tool_names:
            tool = self.get_tool(tool_name)
            if tool:
                pipeline.add_tool(tool)
            else:
                raise ValueError(f"Tool not found: {tool_name}")

        self.pipelines[name] = pipeline
        logger.info(f"Created pipeline: {name} with tools: {tool_names}")

        return pipeline

    def create_research_pipeline(self) -> ToolPipeline:
        """연구용 파이프라인 생성."""
        return self.create_pipeline(
            "research_pipeline",
            ["text_extractor", "json_formatter", "content_filter", "data_aggregator", "validator"]
        )

    def create_data_processing_pipeline(self) -> ToolPipeline:
        """데이터 처리용 파이프라인 생성."""
        return self.create_pipeline(
            "data_processing",
            ["json_formatter", "content_filter", "data_aggregator", "validator"]
        )

    async def execute_tool(self, tool_name: str, input_data: Any, **kwargs) -> ProcessingResult:
        """단일 도구 실행."""
        tool = self.get_tool(tool_name)
        if not tool:
            return ProcessingResult(
                success=False,
                output_data=None,
                processing_time=0.0,
                error_message=f"Tool not found: {tool_name}"
            )

        return await tool.process(input_data, **kwargs)

    async def execute_pipeline(self, pipeline_name: str, input_data: Any, **kwargs) -> Dict[str, Any]:
        """파이프라인 실행."""
        pipeline = self.pipelines.get(pipeline_name)
        if not pipeline:
            return {
                "success": False,
                "error": f"Pipeline not found: {pipeline_name}",
                "total_time": 0.0
            }

        return await pipeline.execute(input_data, **kwargs)

    def get_available_tools(self) -> List[Dict[str, Any]]:
        """사용 가능한 도구 목록."""
        return [
            {
                "name": tool.metadata.name,
                "description": tool.metadata.description,
                "category": tool.metadata.category.value,
                "input_formats": [f.value for f in tool.metadata.input_formats],
                "output_formats": [f.value for f in tool.metadata.output_formats],
                "success_rate": tool.success_rate,
                "usage_count": tool.usage_count,
                "avg_processing_time": tool.avg_processing_time
            }
            for tool in self.tools.values()
        ]

    def get_toolbox_stats(self) -> Dict[str, Any]:
        """도구 모음 통계."""
        total_usage = sum(tool.usage_count for tool in self.tools.values())
        total_success = sum(tool.success_count for tool in self.tools.values())

        category_stats = {}
        for tool in self.tools.values():
            cat = tool.metadata.category.value
            if cat not in category_stats:
                category_stats[cat] = {"count": 0, "usage": 0}
            category_stats[cat]["count"] += 1
            category_stats[cat]["usage"] += tool.usage_count

        return {
            "total_tools": len(self.tools),
            "total_usage": total_usage,
            "overall_success_rate": total_success / total_usage if total_usage > 0 else 0,
            "categories": category_stats,
            "pipelines": list(self.pipelines.keys())
        }


# 전역 Unix 철학 도구 모음 인스턴스
_unix_toolbox = None

def get_unix_toolbox() -> UnixPhilosophyToolbox:
    """전역 Unix 철학 도구 모음 인스턴스 반환."""
    global _unix_toolbox
    if _unix_toolbox is None:
        _unix_toolbox = UnixPhilosophyToolbox()
    return _unix_toolbox

def set_unix_toolbox(toolbox: UnixPhilosophyToolbox):
    """전역 Unix 철학 도구 모음 설정."""
    global _unix_toolbox
    _unix_toolbox = _unix_toolbox


# 편의 함수들
async def quick_extract_text(text: str, pattern: str = r"(.*)") -> ProcessingResult:
    """빠른 텍스트 추출."""
    toolbox = get_unix_toolbox()
    return await toolbox.execute_tool("text_extractor", text, pattern=pattern)

async def quick_format_json(data: Any) -> ProcessingResult:
    """빠른 JSON 포맷팅."""
    toolbox = get_unix_toolbox()
    return await toolbox.execute_tool("json_formatter", data)

async def quick_filter_content(data: Any, condition: str) -> ProcessingResult:
    """빠른 컨텐츠 필터링."""
    toolbox = get_unix_toolbox()
    return await toolbox.execute_tool("content_filter", data, condition=condition)

async def quick_aggregate_data(data: Any) -> ProcessingResult:
    """빠른 데이터 집계."""
    toolbox = get_unix_toolbox()
    return await toolbox.execute_tool("data_aggregator", data)

async def quick_validate_data(data: Any, **kwargs) -> ProcessingResult:
    """빠른 데이터 검증."""
    toolbox = get_unix_toolbox()
    return await toolbox.execute_tool("validator", data, **kwargs)

async def run_research_pipeline(input_data: Any) -> Dict[str, Any]:
    """연구 파이프라인 실행."""
    toolbox = get_unix_toolbox()
    pipeline = toolbox.create_research_pipeline()
    return await toolbox.execute_pipeline("research_pipeline", input_data)
