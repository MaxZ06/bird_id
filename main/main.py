"""Local RA-ViT desktop classifier. Run with: python main/main.py.

Requires PyQt6, Pillow, torch, and torchvision. Checkpoints must contain a
complete RA-ViT state_dict (plain or under model_state_dict). No downloads or
network services are used. Predictions use softmax(global_logits + local_logits).
"""

import argparse
from collections.abc import Mapping
from pathlib import Path
import sys

import torch
from PIL import Image, ImageOps
from PIL.ImageQt import ImageQt
from PyQt6.QtCore import QPointF, QRectF, Qt, QThread, pyqtSignal
from PyQt6.QtGui import QColor, QPainter, QPainterPath, QPen, QPixmap
from PyQt6.QtWidgets import (
    QApplication, QFrame, QHBoxLayout, QLabel, QFileDialog, QMainWindow,
    QProgressBar, QPushButton, QSizePolicy, QVBoxLayout, QWidget,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data_preprocessing.data_splitting import (
    CROPPED_DATA_ROOT, vit_b16_eval_transform,
)
from src.primary_model.models import LOCAL_CROP_SIZE, RA_ViT, VIT_B16_IMAGE_SIZE

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
PREDICTION_PANEL_WIDTH = 520


def read_class_names(classes_path=None):
    """Match ImageFolder's sorted class order, or read CUB's classes.txt."""
    if classes_path is None and CROPPED_DATA_ROOT.is_dir():
        names = sorted(path.name for path in CROPPED_DATA_ROOT.iterdir() if path.is_dir())
        if names:
            return names
    path = Path(classes_path) if classes_path else REPO_ROOT / "CUB_200_2011" / "classes.txt"
    if not path.is_file():
        raise FileNotFoundError(
            "Class names are missing. Keep the cropped dataset or CUB classes.txt "
            "in the project, or launch with --classes path/to/classes.txt."
        )
    names = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            parts = line.strip().split(maxsplit=1)
            names.append(parts[1] if len(parts) == 2 and parts[0].isdigit() else line.strip())
    return names


def load_classifier(checkpoint_path, device, classes_path=None, crop_size=LOCAL_CROP_SIZE):
    """Load weights safely on CPU and infer the classifier dimensions."""
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, Mapping):
        raise ValueError("Select an RA-ViT state-dictionary checkpoint.")
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    if not isinstance(state_dict, Mapping):
        raise ValueError("The checkpoint does not contain a model state dictionary.")
    try:
        num_classes, hidden_dim = state_dict["global_classifier.4.weight"].shape
    except (KeyError, AttributeError, ValueError) as error:
        raise ValueError(
            "This is not a supported RA-ViT checkpoint. ResNet and other "
            "architectures do not provide the combined global/local output."
        ) from error
    class_names = checkpoint.get("class_names") if classes_path is None else None
    if class_names is None:
        class_names = read_class_names(classes_path)
    if (
        not isinstance(class_names, (list, tuple))
        or len(class_names) != num_classes
        or num_classes < 3
        or not all(isinstance(name, str) and name.strip() for name in class_names)
    ):
        raise ValueError(f"The checkpoint requires exactly {num_classes} ordered class names (at least 3).")
    # All backbone parameters are in the checkpoint; never fetch pretrained weights.
    model = RA_ViT(
        num_classes=num_classes, fc1_dim=hidden_dim,
        local_crop_size=crop_size, backbone_weights=None,
    )
    try:
        model.load_state_dict(state_dict, strict=True)
    except RuntimeError as error:
        raise ValueError("Checkpoint parameters do not match the RA-ViT architecture.") from error
    model.to(device)
    model.eval()
    return model, list(class_names)


def read_image(path):
    """Decode once and respect camera orientation before preview and inference."""
    with Image.open(path) as source:
        return ImageOps.exif_transpose(source).convert("RGB")


def prepare_model_image(image, crop_box=None):
    """Crop original pixels in memory, then use the model's evaluation resize."""
    if crop_box is not None:
        left, top, right, bottom = crop_box
        if not (
            0 <= left < right <= image.width
            and 0 <= top < bottom <= image.height
            and right - left == bottom - top
        ):
            raise ValueError("The crop must be a nonempty square inside the original image.")
        image = image.crop(crop_box)
    return vit_b16_eval_transform.transforms[0](image)


def predict_top_three(model, image, class_names, device, crop_box=None):
    with torch.inference_mode():
        # Crop the original image before evaluation resizing and normalization.
        images = prepare_model_image(image, crop_box)
        for transform in vit_b16_eval_transform.transforms[1:]:
            images = transform(images)
        images = images.unsqueeze(0).to(device)
        global_logits, local_logits = model(images, return_branch_logits=True)
        probabilities = torch.softmax(global_logits + local_logits, dim=1)
        scores, indices = probabilities.topk(3, dim=1)
    return [
        (class_names[index], score)
        for index, score in zip(indices[0].cpu().tolist(), scores[0].cpu().tolist())
    ]


class TaskThread(QThread):
    """Keep all widget access on the GUI thread."""

    def __init__(self, operation, parent=None):
        super().__init__(parent)
        self.operation = operation
        self.result = None
        self.error = None

    def run(self):
        try:
            self.result = self.operation()
        except Exception as error:
            self.error = str(error) or type(error).__name__


class ImageDropArea(QLabel):
    file_dropped = pyqtSignal(str)
    selection_changed = pyqtSignal()

    def __init__(self):
        super().__init__("Drop a bird image here\nor use Choose image above")
        self.setObjectName("dropArea")
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setMinimumSize(240, 160)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setAcceptDrops(True)
        self.setMouseTracking(True)
        self.preview = None
        self.original_size = None
        self.crop_enabled = False
        self.crop_box = None
        self.drag_anchor = None
        self.drag_mode = None
        self.drag_box = None

    def dropped_path(self, mime_data):
        if not self.isEnabled() or not mime_data.hasUrls():
            return None
        urls = mime_data.urls()
        if len(urls) != 1 or not urls[0].isLocalFile():
            return None
        path = Path(urls[0].toLocalFile())
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            return str(path)
        return None

    def dragEnterEvent(self, event):
        if self.dropped_path(event.mimeData()):
            event.setDropAction(Qt.DropAction.CopyAction)
            event.accept()
        else:
            event.ignore()

    def dropEvent(self, event):
        path = self.dropped_path(event.mimeData())
        if path:
            event.setDropAction(Qt.DropAction.CopyAction)
            event.accept()
            self.file_dropped.emit(path)
        else:
            event.ignore()

    def show_image(self, image):
        preview = image.copy()
        preview.thumbnail((1400, 1000))
        # Detach from ImageQt's Python-owned pixel buffer before it is released.
        self.preview = QPixmap.fromImage(ImageQt(preview).copy())
        self.setText("")
        self.original_size = image.size
        self.crop_enabled = False
        self.crop_box = None
        self.drag_anchor = None
        self.drag_mode = None
        self.drag_box = None
        self.setCursor(Qt.CursorShape.ArrowCursor)
        self.refresh_preview()

    def refresh_preview(self):
        self.update()

    def image_rect(self):
        """The painted image bounds, excluding padding and letterbox margins."""
        if self.original_size is None:
            return QRectF()
        available = QRectF(self.contentsRect()).adjusted(12, 12, -12, -12)
        width, height = self.original_size
        scale = min(available.width() / width, available.height() / height)
        size = QPointF(width * scale, height * scale)
        return QRectF(available.center() - size / 2, available.center() + size / 2)

    def image_point(self, position):
        rect = self.image_rect()
        width, height = self.original_size
        return (
            min(width, max(0, (position.x() - rect.left()) * width / rect.width())),
            min(height, max(0, (position.y() - rect.top()) * height / rect.height())),
        )

    def set_crop_enabled(self, enabled):
        self.crop_enabled = bool(enabled and self.original_size)
        self.drag_anchor = None
        self.drag_mode = None
        self.drag_box = None
        self.setCursor(Qt.CursorShape.SizeAllCursor if self.crop_enabled else Qt.CursorShape.ArrowCursor)
        if self.crop_enabled:
            self.reset_crop()
        else:
            self.crop_box = None
            self.selection_changed.emit()
            self.update()

    def reset_crop(self):
        """Restore the largest centered square, always relative to the original."""
        if not self.crop_enabled:
            return
        self.drag_anchor = self.drag_mode = self.drag_box = None
        width, height = self.original_size
        side = min(width, height)
        left, top = (width - side) // 2, (height - side) // 2
        self.crop_box = (left, top, left + side, top + side)
        self.selection_changed.emit()
        self.update()

    def selection_rect(self):
        if self.crop_box is None:
            return QRectF()
        rect = self.image_rect()
        left, top, right, bottom = self.crop_box
        scale = rect.width() / self.original_size[0]
        return QRectF(rect.left() + left * scale, rect.top() + top * scale,
                      (right - left) * scale, (bottom - top) * scale)

    def corner_positions(self):
        rect = self.selection_rect()
        return {"nw": rect.topLeft(), "ne": rect.topRight(),
                "sw": rect.bottomLeft(), "se": rect.bottomRight()}

    def hit_test(self, position):
        if not self.isEnabled() or not self.crop_enabled or self.crop_box is None:
            return None
        # Handles take priority over moving; choose the nearest for small crops.
        corners = self.corner_positions()
        nearest = min(corners, key=lambda name: (corners[name] - position).manhattanLength())
        offset = corners[nearest] - position
        if abs(offset.x()) <= 9 and abs(offset.y()) <= 9:
            return nearest
        if self.selection_rect().contains(position):
            return "move"
        return None

    def update_cursor(self, position):
        mode = self.drag_mode or self.hit_test(position)
        cursor = {
            "move": Qt.CursorShape.SizeAllCursor,
            "nw": Qt.CursorShape.SizeFDiagCursor, "se": Qt.CursorShape.SizeFDiagCursor,
            "ne": Qt.CursorShape.SizeBDiagCursor, "sw": Qt.CursorShape.SizeBDiagCursor,
        }.get(mode, Qt.CursorShape.ArrowCursor)
        self.setCursor(cursor)

    def paintEvent(self, event):
        # Let QLabel paint its stylesheet frame, but paint the photo ourselves so
        # selection mapping and rendering use exactly the same rectangle.
        if self.preview is None:
            super().paintEvent(event)
            return
        super().paintEvent(event)
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        rect = self.image_rect()
        painter.drawPixmap(rect, self.preview, QRectF(self.preview.rect()))
        if self.crop_enabled and self.crop_box is not None:
            selection = self.selection_rect()
            shade = QPainterPath()
            shade.addRect(rect)
            shade.addRect(selection)
            painter.fillPath(shade, QColor(0, 0, 0, 125))
            painter.setPen(QPen(QColor("white"), 2))
            painter.drawRect(selection)
            painter.setPen(QPen(QColor("#286448"), 1))
            painter.drawRect(selection)
            painter.setBrush(QColor("white"))
            for corner in self.corner_positions().values():
                painter.drawRect(QRectF(corner.x() - 5, corner.y() - 5, 10, 10))
        painter.end()

    def mousePressEvent(self, event):
        mode = self.hit_test(event.position())
        if mode is not None and event.button() == Qt.MouseButton.LeftButton:
            self.drag_anchor = self.image_point(event.position())
            self.drag_mode = mode
            self.drag_box = self.crop_box
            self.update_cursor(event.position())
            event.accept()
        else:
            super().mousePressEvent(event)

    def update_drag(self, position):
        width, height = self.original_size
        end_x, end_y = self.image_point(position)
        dx, dy = end_x - self.drag_anchor[0], end_y - self.drag_anchor[1]
        left, top, right, bottom = self.drag_box
        side = right - left
        scale = self.image_rect().width() / width
        if self.drag_mode == "move":
            left = min(width - side, max(0, left + round(dx)))
            top = min(height - side, max(0, top + round(dy)))
        else:
            # Resize about the opposite corner, projecting the pointer motion
            # onto the square's diagonal so the aspect ratio never changes.
            sx = -1 if "w" in self.drag_mode else 1
            sy = -1 if "n" in self.drag_mode else 1
            anchor_x = right if sx < 0 else left
            anchor_y = bottom if sy < 0 else top
            maximum = min(anchor_x if sx < 0 else width - anchor_x,
                          anchor_y if sy < 0 else height - anchor_y)
            minimum = min(side, max(1, round(24 / scale)))
            side = min(maximum, max(minimum, round(side + (sx * dx + sy * dy) / 2)))
            left = anchor_x - side if sx < 0 else anchor_x
            top = anchor_y - side if sy < 0 else anchor_y
        box = (left, top, left + side, top + side)
        if box != self.crop_box:
            self.crop_box = box
            self.selection_changed.emit()
            self.update()

    def mouseMoveEvent(self, event):
        if self.drag_anchor is not None and self.isEnabled():
            self.update_drag(event.position())
            event.accept()
        else:
            self.update_cursor(event.position())
            super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if self.drag_anchor is not None and event.button() == Qt.MouseButton.LeftButton:
            self.update_drag(event.position())
            self.drag_anchor = self.drag_mode = self.drag_box = None
            self.update_cursor(event.position())
            event.accept()
        else:
            super().mouseReleaseEvent(event)

    def leaveEvent(self, event):
        if self.drag_mode is None:
            self.setCursor(Qt.CursorShape.ArrowCursor)
        super().leaveEvent(event)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.refresh_preview()


class BirdIdWindow(QMainWindow):
    def __init__(self, device, classes_path=None, crop_size=LOCAL_CROP_SIZE):
        super().__init__()
        self.device = device
        self.classes_path = classes_path
        self.crop_size = crop_size
        self.model = None
        self.class_names = []
        self.image = None
        self.worker = None
        self.on_success = None
        self.setWindowTitle("Bird ID · RA-ViT")
        self.resize(940, 860)
        self.setMinimumSize(680, 760)

        container = QWidget()
        self.setCentralWidget(container)
        layout = QVBoxLayout(container)
        layout.setContentsMargins(24, 16, 24, 16)
        layout.setSpacing(6)
        title = QLabel("Bird ID")
        title.setObjectName("title")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(title)
        subtitle = QLabel("Local bird classification · RA-ViT global + local branches")
        subtitle.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(subtitle)

        inputs = QHBoxLayout()
        inputs.addStretch()
        self.checkpoint_button = QPushButton("Choose checkpoint…")
        self.image_button = QPushButton("Choose image…")
        self.predict_button = QPushButton("Identify bird")
        self.predict_button.setObjectName("primary")
        for button in (self.checkpoint_button, self.image_button, self.predict_button):
            button.setMinimumHeight(42)
            inputs.addWidget(button)
        inputs.addStretch()
        layout.addLayout(inputs)
        self.checkpoint_label = self.centered_label("No checkpoint loaded")
        self.image_label = self.centered_label("No image selected")
        layout.addWidget(self.checkpoint_label)
        layout.addWidget(self.image_label)

        crop_controls = QHBoxLayout()
        crop_controls.addStretch()
        self.crop_button = QPushButton("Square crop")
        self.crop_button.setCheckable(True)
        self.full_image_button = QPushButton("Use full image")
        self.reset_crop_button = QPushButton("Reset")
        self.reset_crop_button.setToolTip("Restore the largest centered square on the original image")
        for button in (self.crop_button, self.full_image_button, self.reset_crop_button):
            crop_controls.addWidget(button)
        crop_controls.addStretch()
        layout.addLayout(crop_controls)
        self.crop_hint = self.centered_label("Optional: enable Square crop, drag inside to move, or drag a corner to resize.")
        layout.addWidget(self.crop_hint)

        self.drop_area = ImageDropArea()
        layout.addWidget(self.drop_area, 1)
        self.crop_dimensions = self.centered_label("Original file stays unchanged")
        layout.addWidget(self.crop_dimensions)
        self.crop_warning = self.centered_label("")
        self.crop_warning.setObjectName("cropWarning")
        self.crop_warning.setMinimumHeight(20)
        layout.addWidget(self.crop_warning)
        self.status = self.centered_label("Choose an RA-ViT checkpoint and a bird image to begin.")
        self.status.setMinimumHeight(36)
        layout.addWidget(self.status)
        self.progress = QProgressBar()
        self.progress.setRange(0, 1)
        self.progress.setValue(0)
        self.progress.setTextVisible(False)
        self.progress.setFixedHeight(4)
        layout.addWidget(self.progress)

        results_title = QLabel("Top 3 combined predictions")
        results_title.setObjectName("sectionTitle")
        results_title.setFixedWidth(PREDICTION_PANEL_WIDTH)
        layout.addWidget(results_title, alignment=Qt.AlignmentFlag.AlignHCenter)
        self.result_labels = []
        for rank in range(1, 4):
            row = QFrame()
            row.setObjectName("resultRow")
            row.setFixedWidth(PREDICTION_PANEL_WIDTH)
            row_layout = QHBoxLayout(row)
            rank_label = QLabel(f"{rank:02d}")
            rank_label.setFixedWidth(32)
            species = QLabel("—")
            species.setTextFormat(Qt.TextFormat.PlainText)
            species.setWordWrap(True)
            score = QLabel("—")
            score.setFixedWidth(85)
            score.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            row_layout.addWidget(rank_label)
            row_layout.addWidget(species, 1)
            row_layout.addWidget(score)
            layout.addWidget(row, alignment=Qt.AlignmentFlag.AlignHCenter)
            self.result_labels.append((species, score))
        note = self.centered_label(
            f"Device: {device} · Scores use softmax(global + local logits); they are not calibrated certainty."
        )
        layout.addWidget(note)

        self.checkpoint_button.clicked.connect(self.choose_checkpoint)
        self.image_button.clicked.connect(self.choose_image)
        self.predict_button.clicked.connect(self.identify)
        self.drop_area.file_dropped.connect(self.select_image)
        self.drop_area.selection_changed.connect(self.crop_changed)
        self.crop_button.toggled.connect(self.drop_area.set_crop_enabled)
        self.full_image_button.clicked.connect(lambda: self.crop_button.setChecked(False))
        self.reset_crop_button.clicked.connect(self.drop_area.reset_crop)
        self.setStyleSheet("""
            QMainWindow, QWidget { background: #f5f7f4; color: #20362a; font-size: 14px; }
            QLabel#title { font-size: 30px; font-weight: 700; }
            QLabel#sectionTitle { font-size: 18px; font-weight: 600; }
            QPushButton { background: white; border: 1px solid #c2cfc4;
                          border-radius: 7px; padding: 8px 16px; }
            QPushButton:hover { background: #e5eee5; }
            QPushButton:checked { background: #d4e8d7; border-color: #286448; }
            QPushButton#primary { background: #286448; color: white; border: none; }
            QPushButton:disabled { background: #e2e6e2; color: #818a81; }
            QPushButton#primary:disabled { background: #e2e6e2; color: #818a81; }
            QLabel#dropArea { background: #eaf0e9; border: 2px dashed #9eaf9f;
                             border-radius: 12px; padding: 10px; color: #536854; }
            QLabel#cropWarning { color: #925400; font-size: 12px; }
            QFrame#resultRow { background: white; border: 1px solid #dce4dc; border-radius: 7px; }
            QFrame#resultRow QLabel { background: transparent; }
            QProgressBar { border: none; background: #dce4dc; }
            QProgressBar::chunk { background: #286448; }
        """)
        self.update_controls()

    @staticmethod
    def centered_label(text):
        label = QLabel(text)
        label.setTextFormat(Qt.TextFormat.PlainText)
        label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        label.setWordWrap(True)
        return label

    def update_controls(self):
        idle = self.worker is None
        self.checkpoint_button.setEnabled(idle)
        self.image_button.setEnabled(idle)
        self.drop_area.setEnabled(idle)
        self.crop_button.setEnabled(idle and self.image is not None)
        self.full_image_button.setEnabled(idle and self.image is not None)
        self.reset_crop_button.setEnabled(idle and self.drop_area.crop_enabled)
        self.predict_button.setEnabled(idle and self.model is not None and self.image is not None)

    def crop_changed(self):
        self.clear_results()
        self.update_controls()
        self.crop_hint.setText(
            "Drag inside the box to move it. Drag a corner handle to resize. Reset centers the square."
            if self.drop_area.crop_enabled else
            "Full image selected. Enable Square crop to avoid stretching a rectangular photo."
        )
        self.status.setText("Input changed. Click Identify bird to run a new prediction.")
        self.update_crop_details()

    def update_crop_details(self):
        if self.image is None:
            return
        box = self.drop_area.crop_box
        width, height = (box[2] - box[0], box[3] - box[1]) if box else self.image.size
        kind = "Crop" if box else "Full image"
        self.crop_dimensions.setText(f"{kind}: {width} × {height} pixels · Original file unchanged")
        if min(width, height) < VIT_B16_IMAGE_SIZE:
            self.crop_warning.setText(
                f"Small {'crop' if box else 'image'}: {width} × {height} pixels. "
                "Enlarging to 224 × 224 cannot restore missing detail."
            )
        else:
            self.crop_warning.setText("")

    def clear_results(self):
        for species, score in self.result_labels:
            species.setText("—")
            score.setText("—")

    def start_task(self, operation, on_success, message):
        if self.worker is not None:
            return
        self.clear_results()
        self.status.setText(message)
        self.progress.setRange(0, 0)
        self.on_success = on_success
        self.worker = TaskThread(operation, self)
        self.worker.finished.connect(self.finish_task)
        self.update_controls()
        self.worker.start()

    def finish_task(self):
        worker = self.worker
        callback = self.on_success
        self.worker = None
        self.on_success = None
        self.progress.setRange(0, 1)
        self.progress.setValue(0)
        try:
            if worker.error:
                self.status.setText(f"Could not complete the operation: {worker.error}")
            else:
                callback(worker.result)
        except Exception as error:
            self.status.setText(f"Could not display the result: {error}")
        finally:
            worker.deleteLater()
            self.update_controls()

    def choose_checkpoint(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select an RA-ViT checkpoint", str(REPO_ROOT / "checkpoints"),
            "All files (*);;PyTorch checkpoints (*.pt *.pth *.ckpt)",
        )
        if path:
            self.select_checkpoint(path)

    def select_checkpoint(self, path):
        def loaded(result):
            self.model, self.class_names = result
            self.checkpoint_label.setText(f"Model: {Path(path).name}")
            self.checkpoint_label.setToolTip(str(path))
            self.status.setText("Model ready. Choose an image, then click Identify bird.")

        self.start_task(
            lambda: load_classifier(path, self.device, self.classes_path, self.crop_size),
            loaded, "Loading checkpoint… This can take a moment.",
        )

    def choose_image(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select a bird image", str(REPO_ROOT),
            "Images (*.jpg *.jpeg *.png *.bmp *.webp *.tif *.tiff);;All files (*)",
        )
        if path:
            self.select_image(path)

    def select_image(self, path):
        def loaded(image):
            self.drop_area.show_image(image)
            self.image = image
            self.crop_button.setChecked(False)
            self.update_crop_details()
            self.crop_hint.setText("Optional: enable Square crop, drag inside to move, or drag a corner to resize.")
            self.image_label.setText(f"Image: {Path(path).name}")
            self.image_label.setToolTip(str(path))
            self.status.setText(
                "Image ready. Click Identify bird." if self.model is not None
                else "Image ready. Choose an RA-ViT checkpoint."
            )

        self.start_task(lambda: read_image(path), loaded, "Opening image…")

    def identify(self):
        if self.worker is not None or self.model is None or self.image is None:
            return
        image, crop_box = self.image, self.drop_area.crop_box
        self.start_task(
            lambda: predict_top_three(self.model, image, self.class_names, self.device, crop_box),
            self.show_predictions, "Identifying bird…",
        )

    def show_predictions(self, predictions):
        for (species_label, score_label), (name, score) in zip(self.result_labels, predictions):
            prefix, separator, species = name.partition(".")
            display_name = species if separator and prefix.isdigit() else name
            species_label.setText(display_name.replace("_", " "))
            score_label.setText(f"{score:.2%}")
        self.status.setText("Prediction complete. Choose or drop another image to try again.")

    def closeEvent(self, event):
        if self.worker is not None:
            self.status.setText("Please wait for the current operation to finish, then close the window.")
            event.ignore()
        else:
            event.accept()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--classes", type=Path, help="Class names in model-index order, one per line.")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    parser.add_argument("--crop-size", type=int, default=LOCAL_CROP_SIZE,
                        help="Local crop size in pixels; must match training (default: 112).")
    args = parser.parse_args()
    if args.crop_size not in range(16, 225, 16):
        parser.error("--crop-size must be a multiple of 16 between 16 and 224.")
    device_name = args.device
    if device_name == "auto":
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
    if device_name == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA is not available. Use --device cpu.")
    if device_name == "mps" and not torch.backends.mps.is_available():
        parser.error("MPS is not available. Use --device cpu.")
    app = QApplication(sys.argv[:1])
    window = BirdIdWindow(torch.device(device_name), args.classes, args.crop_size)
    window.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
