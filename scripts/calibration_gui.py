#!/usr/bin/env python3
"""Qt calibration UI for dual Raspberry Pi cameras."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, Tuple

import cv2
import numpy as np
import yaml
from picamera2 import Picamera2
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

# Ensure we can reuse helper utilities (optional, fall back gracefully)
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

CONFIG_PATH = PROJECT_ROOT / "config" / "camera_profiles.yaml"


def load_profiles() -> Dict[str, dict]:
    if not CONFIG_PATH.exists():
        return {}
    return yaml.safe_load(CONFIG_PATH.read_text()).get("profiles", {})


def save_profiles(profiles: Dict[str, dict]) -> None:
    data = yaml.safe_load(CONFIG_PATH.read_text()) if CONFIG_PATH.exists() else {}
    data.setdefault("profiles", {}).update(profiles)
    CONFIG_PATH.write_text(yaml.safe_dump(data, sort_keys=False))


def cv_rotation_code(angle: int) -> int | None:
    mapping = {0: None, 90: cv2.ROTATE_90_CLOCKWISE, 180: cv2.ROTATE_180, 270: cv2.ROTATE_90_COUNTERCLOCKWISE}
    return mapping.get(angle % 360)


class CameraPanel(QWidget):
    def __init__(self, camera_index: int, profile_key: str, parent: QWidget | None = None):
        super().__init__(parent)
        self.camera_index = camera_index
        self.profile_key = profile_key
        self.profiles = load_profiles()
        profile = self.profiles.get(profile_key, {})
        self.last_processed: np.ndarray | None = None

        self.source_resolution = tuple(profile.get("resolution", [1536, 864]))
        self.rotation = profile.get("rotation", 0)
        self.hflip = profile.get("hflip", False)
        self.vflip = profile.get("vflip", False)
        crop = profile.get("crop", [self.source_resolution[0], self.source_resolution[1]])
        self.crop_width = crop[0]
        self.crop_height = crop[1]
        self.offset_x = profile.get("offset_x", 0)
        self.offset_y = profile.get("offset_y", 0)

        self.picam = Picamera2(camera_num=camera_index)
        config = self.picam.create_video_configuration(
            main={"size": self.source_resolution},
            controls={"FrameRate": 30},
            buffer_count=4,
        )
        self.picam.configure(config)
        self.picam.start()

        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setMinimumSize(640, 360)

        self.rotation_combo = QComboBox()
        for angle in (0, 90, 180, 270):
            self.rotation_combo.addItem(f"{angle}°", angle)
        self.rotation_combo.setCurrentText(f"{self.rotation}°")
        self.rotation_combo.currentIndexChanged.connect(self.on_rotation_changed)

        self.hflip_box = QCheckBox("Horizontal Flip")
        self.hflip_box.setChecked(self.hflip)
        self.hflip_box.toggled.connect(self.on_hflip_toggled)

        self.vflip_box = QCheckBox("Vertical Flip")
        self.vflip_box.setChecked(self.vflip)
        self.vflip_box.toggled.connect(self.on_vflip_toggled)

        self.crop_w_spin = QSpinBox()
        self.crop_w_spin.setRange(64, self.source_resolution[0])
        self.crop_w_spin.setValue(self.crop_width)
        self.crop_w_spin.valueChanged.connect(self.on_crop_changed)

        self.crop_h_spin = QSpinBox()
        self.crop_h_spin.setRange(64, self.source_resolution[1])
        self.crop_h_spin.setValue(self.crop_height)
        self.crop_h_spin.valueChanged.connect(self.on_crop_changed)

        max_offset_x = max(0, self.source_resolution[0] - self.crop_width)
        max_offset_y = max(0, self.source_resolution[1] - self.crop_height)
        self.offset_x_spin = QSpinBox()
        self.offset_x_spin.setRange(-max_offset_x // 2, max_offset_x // 2)
        self.offset_x_spin.setValue(self.offset_x)
        self.offset_x_spin.valueChanged.connect(self.on_offset_changed)

        self.offset_y_spin = QSpinBox()
        self.offset_y_spin.setRange(-max_offset_y // 2, max_offset_y // 2)
        self.offset_y_spin.setValue(self.offset_y)
        self.offset_y_spin.valueChanged.connect(self.on_offset_changed)

        controls_layout = QGridLayout()
        controls_layout.addWidget(QLabel("Rotation"), 0, 0)
        controls_layout.addWidget(self.rotation_combo, 0, 1)
        controls_layout.addWidget(self.hflip_box, 1, 0, 1, 2)
        controls_layout.addWidget(self.vflip_box, 2, 0, 1, 2)
        controls_layout.addWidget(QLabel("Crop W"), 3, 0)
        controls_layout.addWidget(self.crop_w_spin, 3, 1)
        controls_layout.addWidget(QLabel("Crop H"), 4, 0)
        controls_layout.addWidget(self.crop_h_spin, 4, 1)
        controls_layout.addWidget(QLabel("Offset X"), 5, 0)
        controls_layout.addWidget(self.offset_x_spin, 5, 1)
        controls_layout.addWidget(QLabel("Offset Y"), 6, 0)
        controls_layout.addWidget(self.offset_y_spin, 6, 1)

        group = QGroupBox(f"Camera {camera_index} ({profile_key})")
        group_layout = QVBoxLayout(group)
        group_layout.addWidget(self.image_label)
        group_layout.addLayout(controls_layout)

        layout = QVBoxLayout(self)
        layout.addWidget(group)

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_view)
        self.timer.start(50)  # ~20 FPS preview

    def on_rotation_changed(self) -> None:
        self.rotation = self.rotation_combo.currentData()

    def on_hflip_toggled(self, state: bool) -> None:
        self.hflip = state

    def on_vflip_toggled(self, state: bool) -> None:
        self.vflip = state

    def on_crop_changed(self) -> None:
        self.crop_width = self.crop_w_spin.value()
        self.crop_height = self.crop_h_spin.value()
        max_offset_x = max(0, self.source_resolution[0] - self.crop_width)
        max_offset_y = max(0, self.source_resolution[1] - self.crop_height)
        self.offset_x_spin.setRange(-max_offset_x // 2, max_offset_x // 2)
        self.offset_y_spin.setRange(-max_offset_y // 2, max_offset_y // 2)

    def on_offset_changed(self) -> None:
        self.offset_x = self.offset_x_spin.value()
        self.offset_y = self.offset_y_spin.value()

    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        if frame is None:
            return np.zeros((self.source_resolution[1], self.source_resolution[0], 3), dtype=np.uint8)

        max_x0 = max(0, self.source_resolution[0] - self.crop_width)
        max_y0 = max(0, self.source_resolution[1] - self.crop_height)
        x0_nominal = self.source_resolution[0] // 2 - self.crop_width // 2 + self.offset_x
        y0_nominal = self.source_resolution[1] // 2 - self.crop_height // 2 + self.offset_y
        x0 = int(np.clip(x0_nominal, 0, max_x0))
        y0 = int(np.clip(y0_nominal, 0, max_y0))
        x1 = int(np.clip(x0 + self.crop_width, 0, self.source_resolution[0]))
        y1 = int(np.clip(y0 + self.crop_height, 0, self.source_resolution[1]))
        x0 = max(0, x1 - self.crop_width)
        y0 = max(0, y1 - self.crop_height)
        cropped = frame[y0:y1, x0:x1].copy()

        if self.hflip:
            cropped = cv2.flip(cropped, 1)
        if self.vflip:
            cropped = cv2.flip(cropped, 0)

        rot_code = cv_rotation_code(self.rotation)
        if rot_code is not None:
            cropped = cv2.rotate(cropped, rot_code)

        h, w = cropped.shape[:2]
        side = min(h, w)
        if side <= 0:
            return cropped
        y_start = (h - side) // 2
        x_start = (w - side) // 2
        return cropped[y_start : y_start + side, x_start : x_start + side].copy()

    def update_view(self) -> None:
        frame = self.picam.capture_array()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        processed = self.process_frame(frame)
        self.last_processed = processed
        h, w, ch = processed.shape
        image = QImage(processed.data, w, h, ch * w, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(image).scaled(
            self.image_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
        )
        self.image_label.setPixmap(pixmap)

    def shutdown(self) -> None:
        self.timer.stop()
        self.picam.close()

    def serialize(self) -> dict:
        return {
            "description": self.profiles.get(self.profile_key, {}).get("description", ""),
            "resolution": list(self.source_resolution),
            "rotation": int(self.rotation),
            "hflip": bool(self.hflip),
            "vflip": bool(self.vflip),
            "crop": [int(self.crop_width), int(self.crop_height)],
            "offset_x": int(self.offset_x),
            "offset_y": int(self.offset_y),
        }


class CalibrationWindow(QWidget):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Dual Camera Calibration")
        self.swap_order = False
        self.left_panel = CameraPanel(0, "cam0")
        self.right_panel = CameraPanel(1, "cam1")

        self.panels_layout = QHBoxLayout()
        self.panels_layout.addWidget(self.left_panel)
        self.panels_layout.addWidget(self.right_panel)

        self.combined_label = QLabel()
        self.combined_label.setAlignment(Qt.AlignCenter)
        self.combined_label.setMinimumHeight(320)

        self.save_button = QPushButton("Save Calibration")
        self.save_button.clicked.connect(self.save_profiles)

        self.swap_check = QCheckBox("Swap Left/Right Preview")
        self.swap_check.toggled.connect(self.on_swap_toggled)

        controls_layout = QHBoxLayout()
        controls_layout.addWidget(self.swap_check)
        controls_layout.addStretch()
        controls_layout.addWidget(self.save_button)

        root_layout = QVBoxLayout(self)
        root_layout.addLayout(self.panels_layout)
        root_layout.addWidget(self.combined_label)
        root_layout.addLayout(controls_layout)

        self.combined_timer = QTimer(self)
        self.combined_timer.timeout.connect(self.update_combined_view)
        self.combined_timer.start(100)

    def save_profiles(self) -> None:
        profiles = {
            self.left_panel.profile_key: self.left_panel.serialize(),
            self.right_panel.profile_key: self.right_panel.serialize(),
        }
        save_profiles(profiles)
        self.save_button.setText("Saved!")
        QTimer.singleShot(1500, lambda: self.save_button.setText("Save Calibration"))

    def on_swap_toggled(self, checked: bool) -> None:
        self.swap_order = checked
        while self.panels_layout.count():
            item = self.panels_layout.takeAt(0)
            widget = item.widget()
            if widget:
                widget.setParent(None)
        if self.swap_order:
            self.panels_layout.addWidget(self.right_panel)
            self.panels_layout.addWidget(self.left_panel)
        else:
            self.panels_layout.addWidget(self.left_panel)
            self.panels_layout.addWidget(self.right_panel)

    def update_combined_view(self) -> None:
        left_frame = self.left_panel.last_processed
        right_frame = self.right_panel.last_processed
        if left_frame is None or right_frame is None:
            return

        if self.swap_order:
            left_frame, right_frame = right_frame, left_frame

        side = min(left_frame.shape[0], right_frame.shape[0])
        if side <= 0:
            return
        if left_frame.shape[0] != side or left_frame.shape[1] != side:
            left_frame = cv2.resize(left_frame, (side, side), interpolation=cv2.INTER_AREA)
        if right_frame.shape[0] != side or right_frame.shape[1] != side:
            right_frame = cv2.resize(right_frame, (side, side), interpolation=cv2.INTER_AREA)

        combined = np.hstack([left_frame, right_frame])
        h, w, ch = combined.shape
        image = QImage(combined.data, w, h, ch * w, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(image).scaled(
            self.combined_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
        )
        self.combined_label.setPixmap(pixmap)

    def closeEvent(self, event) -> None:  # noqa: N802
        self.combined_timer.stop()
        self.left_panel.shutdown()
        self.right_panel.shutdown()
        super().closeEvent(event)


def main() -> None:
    app = QApplication(sys.argv)
    window = CalibrationWindow()
    window.resize(1400, 700)
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
