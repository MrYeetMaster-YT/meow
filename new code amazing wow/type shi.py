import sys, os, cv2, random, csv, datetime, pickle
import face_recognition
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QPushButton, QLabel, QFileDialog, QWidget,
    QVBoxLayout, QHBoxLayout, QStackedLayout, QTextEdit, QMessageBox
)
from PyQt5.QtCore import Qt, QTimer, QPoint
from PyQt5.QtGui import QImage, QPixmap, QPainter, QColor, QPen

class FaceVaultUltra(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("FaceVault Ultra - PyQt Edition")
        self.setGeometry(100, 100, 1180, 760)

        self.data_dir = "face_data"
        self.log_file = os.path.join(self.data_dir, "logs.csv")
        self.encodings_file = os.path.join(self.data_dir, "encodings.dat")

        os.makedirs(self.data_dir, exist_ok=True)
        self.known_faces = {}
        self.load_encodings()
        self.current_user = None

        self.theme = "dark"
        self.camera_active = False
        self.particles = []
        self.scan_line_y = 0

        self.init_ui()
        self.init_camera()

        self.anim_timer = QTimer()
        self.anim_timer.timeout.connect(self.update_particles)
        self.anim_timer.start(50)

    def init_ui(self):
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)

        layout = QHBoxLayout(self.central_widget)

        # Sidebar
        self.sidebar = QVBoxLayout()
        sidebar_widget = QWidget()
        sidebar_widget.setStyleSheet("background-color: #1f1f2e;")
        sidebar_widget.setFixedWidth(220)
        sidebar_widget.setLayout(self.sidebar)

        self.sidebar.addWidget(self.make_button("🧍 Scan", self.start_camera))
        self.sidebar.addWidget(self.make_button("📊 Dashboard", self.show_dashboard))
        self.sidebar.addWidget(self.make_button("🎨 Toggle Theme", self.toggle_theme))
        self.sidebar.addWidget(self.make_button("❌ Exit", self.close))
        self.sidebar.addStretch()

        # Main Panel
        self.stack = QStackedLayout()
        self.camera_label = QLabel()
        self.camera_label.setFixedSize(860, 480)
        self.camera_label.setStyleSheet("background-color: black;")

        self.status_label = QLabel("🔎 Awaiting scan...")
        self.status_label.setStyleSheet("color: #00f0ff; font-size: 16px;")

        cam_layout = QVBoxLayout()
        cam_layout.addWidget(self.camera_label)
        cam_layout.addWidget(self.status_label)
        cam_widget = QWidget()
        cam_widget.setLayout(cam_layout)

        self.stack.addWidget(cam_widget)

        layout.addWidget(sidebar_widget)
        layout.addLayout(self.stack)

        # Particles
        self.particles = [QPoint(random.randint(0, 1180), random.randint(0, 760)) for _ in range(40)]

    def make_button(self, text, callback):
        btn = QPushButton(text)
        btn.setStyleSheet("color: #00f0ff; background-color: #1f1f2e; padding: 10px; font-size: 14px;")
        btn.clicked.connect(callback)
        return btn

    def toggle_theme(self):
        self.theme = "light" if self.theme == "dark" else "dark"
        self.central_widget.setStyleSheet(
            "background-color: #ffffff;" if self.theme == "light" else "background-color: #0e0e1f;"
        )

    def load_encodings(self):
        if os.path.exists(self.encodings_file):
            with open(self.encodings_file, "rb") as f:
                self.known_faces = pickle.load(f)

    def init_camera(self):
        self.capture = cv2.VideoCapture(0)
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)

    def start_camera(self):
        if not self.camera_active:
            self.timer.start(30)
            self.camera_active = True

    def update_frame(self):
        ret, frame = self.capture.read()
        if not ret: return
        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        face_locations = face_recognition.face_locations(rgb)
        for top, right, bottom, left in face_locations:
            encodings = face_recognition.face_encodings(rgb, [(top, right, bottom, left)])
            label = "Unknown"
            color = (255, 0, 0)
            if encodings:
                for name, known_enc in self.known_faces.items():
                    if face_recognition.compare_faces([known_enc], encodings[0])[0]:
                        label = name
                        color = (0, 255, 0)
                        self.status_label.setText(f"✅ Recognized: {name}")
                        self.log_entry(name)
                        break
            cv2.rectangle(rgb, (left, top), (right, bottom), color, 2)
            cv2.putText(rgb, label, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        self.scan_line_y = (self.scan_line_y + 10) % rgb.shape[0]
        cv2.line(rgb, (0, self.scan_line_y), (rgb.shape[1], self.scan_line_y), (0, 255, 0), 2)

        h, w, ch = rgb.shape
        img = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888)
        self.camera_label.setPixmap(QPixmap.fromImage(img))

    def update_particles(self):
        painter = QPainter(self)
        for i, pt in enumerate(self.particles):
            pt.setY(pt.y() + 2)
            if pt.y() > 760:
                pt.setY(0)
                pt.setX(random.randint(0, 1180))
            painter.setPen(QPen(QColor("#00f0ff"), 2))
            painter.drawPoint(pt)

    def log_entry(self, name):
        now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(self.log_file, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([name, now])

    def show_dashboard(self):
        QMessageBox.information(self, "Dashboard", "Dashboard feature under construction.")

    def closeEvent(self, event):
        self.capture.release()
        event.accept()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = FaceVaultUltra()
    win.show()
    sys.exit(app.exec_())
