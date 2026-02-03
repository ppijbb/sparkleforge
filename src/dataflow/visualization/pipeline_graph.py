"""
Pipeline Graph Visualizer

Pipeline의 데이터 흐름 그래프를 시각화합니다.
"""

import logging
from typing import Optional

from ..pipeline.agent_pipeline import AgentPipeline

logger = logging.getLogger(__name__)


class PipelineGraphVisualizer:
    """
    Pipeline 그래프 시각화 클래스.
    
    DataFlow의 draw_graph() 기능을 참고하여 구현했습니다.
    """
    
    def __init__(self, pipeline: AgentPipeline):
        """
        초기화.
        
        Args:
            pipeline: 시각화할 Pipeline 인스턴스
        """
        self.pipeline = pipeline
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def draw_graph(
        self,
        output_path: Optional[str] = None,
        format: str = "png",
    ) -> str:
        """
        Pipeline 그래프를 그립니다.
        
        Args:
            output_path: 출력 파일 경로 (None이면 기본 경로 사용)
            format: 출력 형식 ("png", "svg", "html")
            
        Returns:
            출력 파일 경로
        """
        if not self.pipeline.compiled:
            self.logger.warning("Pipeline is not compiled. Compiling now...")
            self.pipeline.compile()
        
        try:
            import networkx as nx
        except ImportError:
            raise ImportError(
                "Please install networkx to draw graph. "
                "Run: pip install networkx[default]"
            )
        
        # 그래프 생성
        G = nx.DiGraph()
        
        # 노드 추가
        for op_node in self.pipeline.op_nodes_list:
            node_label = self._get_node_label(op_node)
            G.add_node(op_node, label=node_label)
        
        # 엣지 추가
        for op_node in self.pipeline.op_nodes_list:
            for output_key_node in op_node.output_keys_nodes.values():
                for ptr_key_node in output_key_node.ptr:
                    target_node = self.pipeline.op_nodes_list[ptr_key_node.index]
                    G.add_edge(op_node, target_node, label=ptr_key_node.key)
        
        # 그래프 그리기
        if format == "html":
            return self._draw_html_graph(G, output_path)
        else:
            return self._draw_image_graph(G, output_path, format)
    
    def _get_node_label(self, op_node) -> str:
        """노드 레이블을 생성합니다."""
        op_class_name = (
            op_node.op_obj.__class__.__name__
            if op_node.op_obj
            else "Storage/No-Op"
        )
        return f"{op_node.op_name}\n<{op_class_name}>"
    
    def _draw_html_graph(self, G, output_path: Optional[str]) -> str:
        """HTML 형식으로 그래프를 그립니다."""
        try:
            from pyvis.network import Network
        except ImportError:
            raise ImportError(
                "Please install pyvis to draw HTML graph. "
                "Run: pip install pyvis"
            )
        
        if output_path is None:
            output_path = "pipeline_graph.html"
        
        net = Network(height="800px", width="100%", directed=True)
        
        # 노드 추가
        for node in G.nodes():
            label = G.nodes[node]["label"]
            net.add_node(str(id(node)), label=label, title=label)
        
        # 엣지 추가
        for edge in G.edges(data=True):
            source_id = str(id(edge[0]))
            target_id = str(id(edge[1]))
            edge_label = edge[2].get("label", "")
            net.add_edge(source_id, target_id, label=edge_label, title=edge_label)
        
        # 그래프 저장
        net.save_graph(output_path)
        self.logger.info(f"Pipeline graph saved to {output_path}")
        
        return output_path
    
    def _draw_image_graph(self, G, output_path: Optional[str], format: str) -> str:
        """이미지 형식으로 그래프를 그립니다."""
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            raise ImportError(
                "Please install matplotlib to draw image graph. "
                "Run: pip install matplotlib"
            )
        
        if output_path is None:
            output_path = f"pipeline_graph.{format}"
        
        # 레이아웃 계산
        pos = nx.spring_layout(G)
        
        # 그래프 그리기
        plt.figure(figsize=(12, 8))
        
        # 노드 레이블
        labels = {node: G.nodes[node]["label"] for node in G.nodes()}
        
        # 그래프 그리기
        nx.draw(
            G,
            pos,
            labels=labels,
            with_labels=True,
            node_size=2000,
            node_shape="s",
            node_color="lightblue",
            edge_color="gray",
            arrows=True,
            font_size=8,
        )
        
        # 엣지 레이블
        edge_labels = nx.get_edge_attributes(G, "label")
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=6)
        
        # 저장
        plt.savefig(output_path, format=format, bbox_inches="tight")
        plt.close()
        
        self.logger.info(f"Pipeline graph saved to {output_path}")
        
        return output_path








