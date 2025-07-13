# filename: facevault_app.py

import cv2
import face_recognition
import tkinter as tk
from tkinter import ttk, messagebox
from PIL import Image, ImageTk, ImageFont, ImageDraw
import pickle, os, datetime, threading
from playsound import playsound

class FaceVault:
    def __init__(self, root):
        self.root = root
        self.root.title("FaceVault 2025")
        self.root.geometry("960x640")
        self.root.resizable(False, False)

        self.data_dir = "face_data"
        self.encodings_file = os.path.join(self.data_dir, "encodings.pkl")
        self.student_file = os.path.join(self.data_dir, "students.pkl")
        self.admin_credentials = {"admin": "admin123"}

        os.makedirs(self.data_dir, exist_ok=True)
        self.known_faces = {}
        self.student_data = {}
        self.load_data()

        self.current_user = None
        self.is_admin = False
        self.cap = None
        self.camera_active = False

        self.setup_ui()

    def load_data(self):
        if os.path.exists(self.encodings_file):
            with open(self.encodings_file, "rb") as f:
                self.known_faces = pickle.load(f)
        if os.path.exists(self.student_file):
            with open(self.student_file, "rb") as f:
                self.student_data = pickle.load(f)

    def save_data(self):
        with open(self.encodings_file, "wb") as f:
            pickle.dump(self.known_faces, f)
        with open(self.student_file, "wb") as f:
            pickle.dump(self.student_data, f)

    def setup_ui(self):
        # Background
        try:
            bg = Image.open("nebula.jpg").resize((960, 640))
            self.bg_image = ImageTk.PhotoImage(bg)
            bg_label = tk.Label(self.root, image=self.bg_image)
        except:
            bg_label = tk.Label(self.root, bg="#111111")
        bg_label.place(x=0, y=0, relwidth=1, relheight=1)

        # Neon Header
        self.header = tk.Label(self.root, text="⚡ FACEVAULT SCANNER ⚡",
                               font=("Orbitron", 26, "bold"),
                               fg="#00ffff", bg="#000000")
        self.header.pack(pady=20)

        # Camera frame
        self.camera_frame = tk.Frame(self.root, width=640, height=480, bg="#111111")
        self.camera_frame.pack()
        self.camera_label = tk.Label(self.camera_frame)
        self.camera_label.pack()

        # Entry
        self.name_var = tk.StringVar()
        self.name_entry = ttk.Entry(self.root, textvariable=self.name_var,
                                    font=("Orbitron", 16), width=30)
        self.name_entry.pack(pady=10)

        # Status
        self.status_var = tk.StringVar(value="🔄 Awaiting action")
        self.status_label = tk.Label(self.root, textvariable=self.status_var,
                                     font=("Orbitron", 14), fg="white", bg="#111111")
        self.status_label.pack(pady=5)

        # Buttons
        btn_frame = tk.Frame(self.root, bg="#111111")
        btn_frame.pack(pady=10)

        ttk.Button(btn_frame, text="🎥 Start Camera", command=self.start_camera).pack(side=tk.LEFT, padx=10)
        ttk.Button(btn_frame, text="🛡️ Admin Login", command=self.admin_login).pack(side=tk.LEFT, padx=10)
        ttk.Button(btn_frame, text="❌ Exit", command=self.root.quit).pack(side=tk.LEFT, padx=10)

    def start_camera(self):
        if not self.camera_active:
            self.cap = cv2.VideoCapture(0)
            if not self.cap.isOpened():
                messagebox.showerror("Camera Error", "Could not access camera.")
                return
            self.camera_active = True
            self.status_var.set("🎥 Camera On")
            self.update_camera()

    def stop_camera(self):
        if self.cap:
            self.cap.release()
        self.camera_active = False
        self.camera_label.config(image='')

    def update_camera(self):
        ret, frame = self.cap.read()
        if ret:
            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            face_locations = face_recognition.face_locations(rgb)

            if face_locations:
                encodings = face_recognition.face_encodings(rgb, face_locations)
                self.process_face(encodings[0], frame)
            img = Image.fromarray(rgb).resize((640, 480))
            img_tk = ImageTk.PhotoImage(img)
            self.camera_label.img = img_tk
            self.camera_label.config(image=img_tk)
        if self.camera_active:
            self.root.after(30, self.update_camera)

    def process_face(self, encoding, frame):
        name = self.name_var.get().strip()
        for known_name, known_enc in self.known_faces.items():
            if face_recognition.compare_faces([known_enc], encoding)[0]:
                self.recognize_user(known_name, frame)
                return
        if name:
            self.register_new_face(name, encoding, frame)

    def recognize_user(self, name, frame):
        self.current_user = name
        self.status_var.set(f"✅ Recognized: {name}")
        self.play_sound("confirmation.wav")
        self.root.after(2000, lambda: self.show_summary(name))
        self.stop_camera()

    def register_new_face(self, name, encoding, frame):
        self.known_faces[name] = encoding
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
        self.student_data[name] = {"timestamp": timestamp}
        cv2.imwrite(os.path.join(self.data_dir, f"{name}.jpg"), frame)
        self.save_data()
        self.status_var.set(f"🆕 Registered: {name}")
        self.play_sound("confirmation.wav")
        self.root.after(2000, lambda: self.show_summary(name))
        self.stop_camera()

    def show_summary(self, name):
        popup = tk.Toplevel(self.root)
        popup.title("🎓 Student Summary")
        popup.geometry("400x500")
        img_path = os.path.join(self.data_dir, f"{name}.jpg")
        try:
            img = Image.open(img_path).resize((200, 200))
            img_tk = ImageTk.PhotoImage(img)
            tk.Label(popup, image=img_tk).pack(pady=10)
            tk.Label(popup, text=f"Name: {name}", font=("Orbitron", 14)).pack()
            tk.Label(popup, text=f"Registered: {self.student_data[name]['timestamp']}", font=("Orbitron", 12)).pack()
        except:
            tk.Label(popup, text="Image unavailable").pack()
        tk.Label(popup, text=f"Total Registered: {len(self.known_faces)}", font=("Orbitron", 12)).pack(pady=10)
        ttk.Button(popup, text="Close", command=popup.destroy).pack(pady=20)

    def admin_login(self):
        login = tk.Toplevel(self.root)
        login.title("Admin Login")
        login.geometry("300x200")

        tk.Label(login, text="Username:", font=("Arial", 12)).pack(pady=5)
        user_entry = ttk.Entry(login)
        user_entry.pack(pady=5)

        tk.Label(login, text="Password:", font=("Arial", 12)).pack(pady=5)
        pass_entry = ttk.Entry(login, show="*")
        pass_entry.pack(pady=5)

        def attempt_login():
            u, p = user_entry.get(), pass_entry.get()
            if self.admin_credentials.get(u) == p:
                self.show_admin_panel()
                login.destroy()
            else:
                messagebox.showerror("Access Denied", "Incorrect credentials")

        ttk.Button(login, text="Login", command=attempt_login).pack(pady=10)

    def show_admin_panel(self):
        admin = tk.Toplevel(self.root)
        admin.title("🛡️ Admin Panel")
        admin.geometry("800x500")

        tree = ttk.Treeview(admin, columns=("Name", "Registered"), show="headings")
        tree.heading("Name", text="Student Name")
        tree.heading("Registered", text="Registration Date")
        tree.pack(fill="both", expand=True)

        for name in self.known_faces:
            timestamp = self.student_data.get(name, {}).get("timestamp", "Unknown")
            tree.insert("", "end", values=(name, timestamp))

        def view_face():
            selected = tree.focus()
            if selected:
                sel_name = tree.item(selected)["values"][0]
                self.view_student_details(sel_name)

        ttk.Button(admin, text="View Face", command=view_face).pack(pady=10)

    def view_student_details(self, name):
        detail = tk.Toplevel(self.root)
        detail.title(f"Details: {name}")
        detail.geometry("400x400")
        try:
            img = Image.open(os.path.join(self.data_dir, f"{name}.jpg")).resize((300, 300))
            img_tk = ImageTk.PhotoImage(img)
            tk.Label(detail, image=img_tk).pack(pady=10)
            tk.Label(detail, text=f"Name: {name}", font=("Orbitron", 14)).pack()
        except:
            tk.Label(detail, text="No image").pack()

    def play_sound(self, file):
        if os.path.exists(file):
            threading.Thread(target=lambda: playsound(file), daemon=True).start()

    def run(self):
        self.root.mainloop()


if __name__ == "__main__":
    root = tk.Tk()
    app = FaceVault(root)
    app.run()
