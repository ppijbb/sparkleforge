import asyncio
import random
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional

@dataclass
class MarsPacket:
    """화성-지구 간 전송되는 데이터 패킷 단위"""
    packet_id: int
    content: str
    timestamp: float = field(default_factory=time.time)
    redundancy_data: str = ""  # FEC용 중복 데이터
    is_fec_enabled: bool = True

class MarsCommProtocol:
    """
    지능형 화성 통신 프로토콜 (DTN/BP 개념 적용)
    - 3분(180s) ~ 22분(1320s)의 지연 시간 처리
    - FEC(Forward Error Correction) 기반 자가 복구 로직
    - 비동기 Non-blocking 전송 아키텍처
    """
    def __init__(self, packet_loss_rate=0.15, fec_efficiency=0.6):
        self.plr = packet_loss_rate  # 패킷 손실률
        self.fec_efficiency = fec_efficiency  # FEC 복구 효율
        self.latency_range = (180, 1320)  # 초 단위 (3분~22분)
        self.buffer: List[MarsPacket] = []
        self.stats = {"sent": 0, "received": 0, "fec_recovered": 0, "lost": 0}

    def _apply_fec(self, packet: MarsPacket):
        """FEC 캡슐화 모사"""
        packet.redundancy_data = f"RECOVERY_HEADER_{packet.packet_id}"
        return packet

    async def transmit_async(self, packet: MarsPacket):
        """비동기 전송 시뮬레이션 (Store-and-Forward)"""
        self.stats["sent"] += 1
        
        # 1. 아키텍처 핵심: 지연 시간 계산 (지수 분포 또는 가변 범위)
        delay = random.uniform(*self.latency_range)
        # 시뮬레이션 가속 (1초 = 1분으로 계산)
        sim_delay = delay / 60 
        
        print(f"[{datetime.now().strftime('%H:%M:%S')}] [EARTH->MARS] Packet {packet.packet_id} sent. Expecting latency: {delay/60:.2f} min.")
        
        await asyncio.sleep(sim_delay)
        
        # 2. 패킷 손실 및 복구 로직
        if random.random() < self.plr:
            if packet.is_fec_enabled and random.random() < self.fec_efficiency:
                self.stats["fec_recovered"] += 1
                self.stats["received"] += 1
                print(f"[{datetime.now().strftime('%H:%M:%S')}] [MARS_NODE] Packet {packet.packet_id} LOST but RECOVERED via FEC.")
            else:
                self.stats["lost"] += 1
                print(f"[{datetime.now().strftime('%H:%M:%S')}] [MARS_NODE] Packet {packet.packet_id} CRITICAL LOSS. Requesting retransmission (Wait 44m RTT...)")
        else:
            self.stats["received"] += 1
            print(f"[{datetime.now().strftime('%H:%M:%S')}] [MARS_NODE] Packet {packet.packet_id} received successfully.")

    async def run_mission_simulation(self, total_packets=10):
        """전체 미션 통신 시나리오 실행"""
        print(f"--- Mars Communication Mission Start (PLR: {self.plr*100}%, FEC: {self.fec_efficiency*100}%) ---")
        tasks = []
        for i in range(total_packets):
            p = MarsPacket(packet_id=i, content=f"Scientific Data Chunk {i}")
            tasks.append(self.transmit_async(self._apply_fec(p)))
            # 패킷 간 전송 간격 (0.5분 시뮬레이션)
            await asyncio.sleep(0.5)
        
        await asyncio.gather(*tasks)
        self.report_stats()

    def report_stats(self):
        print("\n" + "="*40)
        print("MARS MISSION COMMUNICATION REPORT")
        print("="*40)
        print(f"Total Packets Sent:     {self.stats['sent']}")
        print(f"Successfully Received:  {self.stats['received']}")
        print(f"  - Clean Receipt:      {self.stats['received'] - self.stats['fec_recovered']}")
        print(f"  - FEC Recovered:      {self.stats['fec_recovered']}")
        print(f"Critical Lost:          {self.stats['lost']}")
        success_rate = (self.stats['received'] / self.stats['sent']) * 100
        print(f"Final Success Rate:     {success_rate:.2f}%")
        print("="*40)

if __name__ == "__main__":
    protocol = MarsCommProtocol()
    asyncio.run(protocol.run_mission_simulation(15))
