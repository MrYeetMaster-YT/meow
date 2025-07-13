import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import matplotlib.pyplot as plt
from PIL import Image, ImageTk, ImageFilter
import datetime
import csv
import os
import cv2
import face_recognition
import threading
import pickle
import random

class FaceVaultUltra:
    def __init__(self, root):
        self.root = root
        self.root.title("FaceVault Ultra")
        self.root.geometry("1180x760")
        self.root.resizable(False, False)

        self.data_dir = "face_data"
        self.log_file = os.path.join(self.data_dir, "logs.csv")
        self.encodings_file = os.path.join(self.data_dir, "encodings.dat")

        os.makedirs(self.data_dir, exist_ok=True)
        self.known_faces = {}
        self.admin_face_encoding = None
        self.current_user = None

        self.theme = "dark"
        self.particles = []
        self.setup_styles()
        self.load_encodings()
        self.setup_main_ui()

        self.cap = None
        self.camera_active = False
        self.scan_line_y = 0

    def setup_styles(self):
        self.style = ttk.Style()
        self.root.configure(bg="#0e0e1f" if self.theme == "dark" else "#ffffff")
        self.style.theme_use("clam")
        self.style.configure("TButton",
            foreground="#00f0ff",
            background="#1f1f2e",
            font=("Segoe UI", 11),
            padding=6)
        self.style.map("TButton",
            background=[("active", "#2a2a40")])

    def setup_main_ui(self):
        self.clear_window()

        self.canvas_bg = tk.Canvas(self.root, width=1180, height=760, bg="#0e0e1f", highlightthickness=0)
        self.canvas_bg.place(x=0, y=0)
        self.animate_particles()

        sidebar = tk.Frame(self.root, width=220, bg="#1f1f2e")
        sidebar.place(x=0, y=0, height=760)

        tk.Label(sidebar, text="🌌 FACEVAULT ULTRA", font=("Orbitron", 16), fg="#00f0ff", bg="#1f1f2e").pack(pady=25)
        for text, cmd in [
            ("🧍 Scan", self.setup_main_ui),
            ("📊 Dashboard", self.show_dashboard),
            ("🧠 Admin", self.admin_login),
            ("🎨 Toggle Theme", self.toggle_theme),
            ("❌ Exit", self.root.quit)]:
            ttk.Button(sidebar, text=text, command=cmd).pack(fill=tk.X, pady=8, padx=10)

        content = tk.Frame(self.root, bg=self.root["bg"])
        content.place(x=230, y=0, width=940, height=760)

        glass_frame = tk.Frame(content, bg="#1a1a2b")
        glass_frame.place(x=40, y=40, width=860, height=660)

        self.camera_label = tk.Label(glass_frame, bg="black")
        self.camera_label.place(x=0, y=0, width=860, height=480)

        self.name_var = tk.StringVar()
        ttk.Entry(glass_frame, textvariable=self.name_var, width=40).place(x=300, y=490)

        self.status_var = tk.StringVar(value="🔎 Awaiting scan...")
        tk.Label(glass_frame, textvariable=self.status_var, font=("Segoe UI", 12), fg="#00f0ff", bg="#1a1a2b").place(x=300, y=530)

        ttk.Button(glass_frame, text="▶ Start Camera", command=self.start_camera).place(x=300, y=570)
        ttk.Button(glass_frame, text="⏹ Stop Camera", command=self.stop_camera).place(x=440, y=570)

    def animate_particles(self):
        self.particles = [(random.randint(0, 1180), random.randint(0, 760), random.choice(["#00f0ff", "#0055ff"])) for _ in range(40)]
        def move():
            self.canvas_bg.delete("particle")
            new_particles = []
            for x, y, color in self.particles:
                y += 1
                if y > 760:
                    y = 0
                self.canvas_bg.create_oval(x, y, x+2, y+2, fill=color, outline=color, tags="particle")
                new_particles.append((x, y, color))
            self.particles = new_particles
            self.root.after(50, move)
        move()

    def toggle_theme(self):
        self.theme = "light" if self.theme == "dark" else "dark"
        self.setup_styles()
        self.setup_main_ui()

    def clear_window(self):
        for widget in self.root.winfo_children():
            widget.destroy()

    def load_encodings(self):
        if os.path.exists(self.encodings_file):
            with open(self.encodings_file, "rb") as f:
                self.known_faces = pickle.load(f)
                if "admin" in self.known_faces:
                    self.admin_face_encoding = self.known_faces["admin"]

    def start_camera(self):
        self.cap = cv2.VideoCapture(0)
        self.camera_active = True
        self.scan_line_y = 0
        self.update_camera()

    def stop_camera(self):
        if self.cap:
            self.cap.release()
        self.camera_active = False
        self.camera_label.config(image='')

    def update_camera(self):
        ret, frame = self.cap.read()
        if not ret:
            return

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb)

        for (top, right, bottom, left) in face_locations:
            match_found = False
            label = "Unknown"
            encodings = face_recognition.face_encodings(rgb, [ (top, right, bottom, left) ])
            if encodings:
                encoding = encodings[0]
                for known_name, known_enc in self.known_faces.items():
                    if face_recognition.compare_faces([known_enc], encoding)[0]:
                        label = known_name
                        match_found = True
                        self.recognize_user(label)
                        break
            color = (0, 255, 0) if match_found else (255, 0, 0)
            cv2.rectangle(rgb, (left, top), (right, bottom), color, 2)
            cv2.putText(rgb, label, (left, top - 10), cv2.FONT_HERSHEY_DUPLEX, 0.7, color, 2)

        height = rgb.shape[0]
        self.scan_line_y = (self.scan_line_y + 12) % height
        cv2.line(rgb, (0, self.scan_line_y), (rgb.shape[1], self.scan_line_y), (0, 255, 0), 2)

        img = Image.fromarray(rgb)
        img = img.resize((860, 480))
        img_tk = ImageTk.PhotoImage(img)
        self.camera_label.config(image=img_tk)
        self.camera_label.image = img_tk

        if self.camera_active:
            self.root.after(30, self.update_camera)

    def recognize_user(self, name):
        self.current_user = name
        self.status_var.set(f"✅ Recognized: {name}")
        self.log_entry(name)
        self.root.after(1500, self.show_dashboard)
        self.stop_camera()

    def log_entry(self, name):
        now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(self.log_file, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([name, now])

    def show_dashboard(self):
        self.clear_window()
        tk.Label(self.root, text=f"📊 Dashboard - {self.current_user}", font=("Orbitron", 18), fg="#00f0ff", bg=self.root["bg"]).pack(pady=20)

        ttk.Button(self.root, text="📋 Show Logs", command=self.show_logs).pack(pady=10)
        ttk.Button(self.root, text="📈 Attendance Graph", command=self.show_graph).pack(pady=10)
        ttk.Button(self.root, text="⬅ Back", command=self.setup_main_ui).pack(pady=20)

    def show_logs(self):
        if not os.path.exists(self.log_file):
            messagebox.showinfo("Logs", "No logs available yet.")
            return

        top = tk.Toplevel(self.root)
        top.title("Attendance Logs")
        top.geometry("500x400")

        with open(self.log_file, "r") as f:
            lines = f.readlines()

        text = tk.Text(top, wrap="none", font=("Consolas", 10))
        text.pack(expand=True, fill="both")
        for line in lines:
            text.insert("end", line)

    def show_graph(self):
        if not os.path.exists(self.log_file):
            messagebox.showinfo("No data", "No log data to plot.")
            return

        dates = {}
        with open(self.log_file, "r") as f:
            reader = csv.reader(f)
            for row in reader:
                date = row[1].split(" ")[0]
                dates[date] = dates.get(date, 0) + 1

        x = list(dates.keys())
        y = list(dates.values())

        plt.figure(figsize=(6, 4))
        plt.bar(x, y, color="#00f0ff")
        plt.title("Attendance Over Time")
        plt.xlabel("Date")
        plt.ylabel("Entries")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

    def admin_login(self):
        if self.current_user == "admin":
            self.show_dashboard()
        else:
            messagebox.showinfo("Admin", "Access denied. Only admin user can access this.")

if __name__ == "__main__":
    root = tk.Tk()
    app = FaceVaultUltra(root)
    root.mainloop()
