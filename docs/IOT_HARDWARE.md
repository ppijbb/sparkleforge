# IoT 어댑터 실제 하드웨어 연동 가이드

`src/core/actuate/iot_device.py`의 디바이스 어댑터는 세 가지 백엔드를 지원합니다.

| 백엔드 | 동작 |
|---|---|
| `auto` (기본) | 실드라이버 연결을 시도하고, 라이브러리나 디바이스가 없으면 Mock으로 폴백 |
| `hardware` | 실드라이버 필수. 연결 실패 시 `connect()`가 `False` 반환 |
| `mock` | 하드웨어가 있어도 항상 인메모리 Mock 사용 |

연결 후 `device.active_backend`로 실제 사용 중인 백엔드를 확인할 수 있습니다.

## 설치

```bash
pip install "sparkleforge[iot]"
# 또는 개별 설치
pip install pyserial hidapi gpiod   # gpiod는 Linux 전용 (libgpiod v2 바인딩)
```

| 어댑터 | 드라이버 | 비고 |
|---|---|---|
| `SerialDevice` | `pyserial` | RS232/USB-Serial |
| `USBHIDDevice` | `hidapi` (`hid` 모듈) | Linux에서는 udev rule로 권한 필요할 수 있음 |
| `GPIODevice` | `gpiod` (libgpiod v2) | `/dev/gpiochipN` 문자 디바이스 사용, Raspberry Pi 등 |
| `RobotArmDevice` | `pyserial` (SerialDevice 경유) | `SET_JOINT_<id>:<angle>` 텍스트 프로토콜 |

## 사용 예

```python
from src.core.actuate.iot_device import SerialDevice, GPIODevice, USBHIDDevice, RobotArmDevice

# 시리얼: 실물 포트가 있으면 hardware, 없으면 mock 폴백
serial = SerialDevice("thermo", port="/dev/ttyUSB0", baudrate=9600)
serial.connect()
print(serial.active_backend)          # "hardware" 또는 "mock"
print(serial.execute_command("GET_TEMP"))

# GPIO: 하드웨어 필수 모드
gpio = GPIODevice("relay", backend="hardware", chip_path="/dev/gpiochip0")
if gpio.connect():
    gpio.execute_command("set_pin 17 1")

# 로봇팔: 백엔드는 내부 시리얼 어댑터로 전달됨
arm = RobotArmDevice("arm-1", serial_port="/dev/ttyUSB1", backend="auto")
arm.connect()
arm.execute_command("move_joint 2 90")
```

## 실기기 스모크 테스트 절차

CI는 하드웨어가 없으므로 Mock 폴백 경로만 검증합니다. 실기기 검증은 대상 장비에서 아래 절차로 수행하세요.

1. **환경 준비**
   ```bash
   pip install "sparkleforge[iot]"
   # 시리얼/HID 권한 (Debian 계열)
   sudo usermod -aG dialout,plugdev $USER  # 재로그인 필요
   ```
2. **시리얼 루프백 테스트** — TX/RX 핀을 점퍼로 연결(루프백)한 뒤:
   ```python
   s = SerialDevice("loopback", port="/dev/ttyUSB0", backend="hardware")
   assert s.connect() and s.active_backend == "hardware"
   s.write("PING\n")
   assert s.read().strip() == "PING"   # 루프백 에코 확인
   ```
3. **GPIO LED 테스트** — LED(+저항)를 대상 핀에 연결:
   ```python
   g = GPIODevice("led", backend="hardware")
   assert g.connect()
   g.execute_command("set_pin 17 1")   # LED 점등 확인
   g.execute_command("set_pin 17 0")   # LED 소등 확인
   ```
4. **USB HID 테스트** — `lsusb`로 VID/PID 확인 후:
   ```python
   h = USBHIDDevice("pad", vendor_id=0x1234, product_id=0x5678, backend="hardware")
   assert h.connect()
   print(h.read())                     # 입력 리포트 수신 확인
   ```
5. **폴백 검증** — 디바이스를 분리한 상태에서 `backend="auto"`로 연결하면
   `active_backend == "mock"`이어야 하며, `backend="hardware"`는 `connect() == False`여야 합니다.

## 제한 사항

- `GPIODevice` 하드웨어 백엔드는 현재 출력 라인 구동만 지원합니다 (입력 라인 읽기는 후속 작업).
- `SensorDevice`, `CameraDevice`는 아직 Mock 전용입니다. 실물 센서는 `SerialDevice`를 조합해 사용하세요.
