import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import cv2
import numpy as np
import matplotlib
matplotlib.use('TkAgg')  # Гарантируем корректную работу бэкенда
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
from scipy.spatial import Delaunay
from scipy.interpolate import LinearNDInterpolator
from mpl_toolkits.mplot3d import Axes3D
from PIL import Image, ImageTk
import os
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

class ReliefInterpolationApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Интерполяция рельефа - Триангуляция Делоне")
        self.root.geometry("1400x900")
        
        # Инициализация переменных
        self.points_pixels = None
        self.points_meters = None
        self.heights = None
        self.triangles = None
        self.grid_x = None
        self.grid_y = None
        self.grid_heights = None
        self.contour_lines = []
        self.contour_levels = None
        self.scale = None
        self.palette_step_m = None
        self.palette_step_mm = None
        self.original_img = None
        self.is_processed = False
        
        # Настройки
        self.num_contours = 10
        self.transparency_3d = 0.7
        self.temp_points = []
        
        # Создание GUI
        self.setup_gui()

    def setup_gui(self):
        """Создание интерфейса с поддержкой скролла мыши и фиксированной шириной"""
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Левая панель
        left_container = ttk.Frame(main_frame, width=400)
        left_container.pack(side=tk.LEFT, fill=tk.BOTH, padx=(0, 10))
        left_container.pack_propagate(False)
        
        # Canvas для прокрутки
        left_canvas = tk.Canvas(left_container, highlightthickness=0)
        left_scrollbar = ttk.Scrollbar(left_container, orient="vertical", command=left_canvas.yview)
        left_scrollable_frame = ttk.Frame(left_canvas)
        left_scrollable_frame.bind("<Configure>", lambda e: left_canvas.configure(scrollregion=left_canvas.bbox("all")))
        
        left_canvas.create_window((0, 0), window=left_scrollable_frame, anchor="nw", width=380)
        left_canvas.configure(yscrollcommand=left_scrollbar.set)
        left_canvas.pack(side="left", fill="both", expand=True)
        left_scrollbar.pack(side="right", fill="y")
        
        # ИСПРАВЛЕНИЕ 2: Привязка прокрутки колесиком мыши к левой панели
        def on_mousewheel(event):
            left_canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")
        
        # Привязываем ко всем элементам левой панели, чтобы скролл работал в любой её точке
        left_canvas.bind_all("<MouseWheel>", on_mousewheel)
        
        right_panel = ttk.Frame(main_frame)
        right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        left_panel = left_scrollable_frame
        
        # Заголовок
        ttk.Label(left_panel, text="ИНТЕРПОЛЯЦИЯ РЕЛЬЕФА", font=('Arial', 14, 'bold')).pack(pady=(0, 15))
        
        # 1. Загрузка изображения
        ttk.Label(left_panel, text="1. Загрузка изображения", font=('Arial', 12, 'bold')).pack(pady=(0, 5))
        self.btn_load_img = ttk.Button(left_panel, text="📁 Загрузить изображение", command=self.load_image)
        self.btn_load_img.pack(fill=tk.X, pady=5)
        
        self.img_label = ttk.Label(left_panel, text="Изображение не загружено", foreground="gray")
        self.img_label.pack(pady=5)
        
        self.preview_frame = ttk.Frame(left_panel)
        self.preview_frame.pack(pady=5)
        self.img_preview = tk.Label(self.preview_frame, bg='lightgray', width=350, height=250, relief=tk.SUNKEN)
        self.img_preview.pack()
        
        self.points_info_label = ttk.Label(left_panel, text="", foreground="blue", wraplength=380)
        self.points_info_label.pack(pady=5)
        
        ttk.Separator(left_panel, orient='horizontal').pack(fill=tk.X, pady=10)
        
        # ИСПРАВЛЕНИЕ 1: Полное отключение алгоритма распознавания, оставляем только ручной ввод
        ttk.Label(left_panel, text="2. Определение точек", font=('Arial', 12, 'bold')).pack(pady=(0, 5))
        self.btn_manual_mode = tk.Button(
            left_panel, 
            text="🎯 НАЧАТЬ РАЗМЕТКУ КАРТЫ", 
            command=self.enable_manual_mode, 
            bg='#2196F3', 
            fg='white', 
            font=('Arial', 11, 'bold'), 
            height=2, 
            relief=tk.RAISED
        )
        self.btn_manual_mode.pack(fill=tk.X, pady=8)
        
        self.btn_clear = ttk.Button(left_panel, text="🗑️ ОЧИСТИТЬ ВСЕ ТОЧКИ", command=self.clear_all)
        self.btn_clear.pack(fill=tk.X, pady=2)
        
        ttk.Separator(left_panel, orient='horizontal').pack(fill=tk.X, pady=10)
        
        # 3. Данные высот
        ttk.Label(left_panel, text="3. Данные высот", font=('Arial', 12, 'bold')).pack(pady=(0, 5))
        self.btn_heights = ttk.Button(left_panel, text="📝 ВВЕСТИ ВЫСОТЫ ВРУЧНУЮ", command=self.input_heights)
        self.btn_heights.pack(fill=tk.X, pady=3)
        
        self.btn_csv_heights = ttk.Button(left_panel, text="📂 ЗАГРУЗИТЬ ИЗ CSV/TXT", command=self.load_heights_from_csv)
        self.btn_csv_heights.pack(fill=tk.X, pady=3)
        
        ttk.Separator(left_panel, orient='horizontal').pack(fill=tk.X, pady=10)
        
                # 4. Параметры плана
        ttk.Label(left_panel, text="4. Параметры плана", font=('Arial', 12, 'bold')).pack(pady=(0, 5))
        params_frame = ttk.LabelFrame(left_panel, text="Введите параметры")
        params_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(params_frame, text="Масштаб плана (1:X):").grid(row=0, column=0, sticky='w', padx=5, pady=5)
        self.scale_entry = ttk.Entry(params_frame, width=15)
        self.scale_entry.grid(row=0, column=1, padx=5, pady=5)
        self.scale_entry.insert(0, "1000")
        
        ttk.Label(params_frame, text="Шаг палетки (мм):").grid(row=1, column=0, sticky='w', padx=5, pady=5)
        self.step_mm_entry = ttk.Entry(params_frame, width=15)
        self.step_mm_entry.grid(row=1, column=1, padx=5, pady=5)
        self.step_mm_entry.insert(0, "10")

        
        # 5. Построение
        self.btn_process = tk.Button(left_panel, text="▶ ПОСТРОИТЬ РЕЛЬЕФ", command=self.process_data, bg='#4CAF50', fg='white', font=('Arial', 12, 'bold'), height=2, relief=tk.RAISED)
        self.btn_process.pack(fill=tk.X, pady=15)
        
        # 6. Настройки отображения
        ttk.Label(left_panel, text="5. Настройки отображения", font=('Arial', 12, 'bold')).pack(pady=(0, 5))
        self.show_points_var = tk.BooleanVar(value=True)
        self.show_grid_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(left_panel, text="Показать исходные точки", variable=self.show_points_var, command=self.redraw_2d_only).pack(anchor='w', pady=2)
        ttk.Checkbutton(left_panel, text="Показать сетку", variable=self.show_grid_var, command=self.redraw_2d_only).pack(anchor='w', pady=2)
        
        ttk.Separator(left_panel, orient='horizontal').pack(fill=tk.X, pady=10)
        
        # 7. Сохранение
        ttk.Label(left_panel, text="6. Сохранение", font=('Arial', 12, 'bold')).pack(pady=(0, 5))
        btn_frame = ttk.Frame(left_panel)
        btn_frame.pack(fill=tk.X, pady=5)
        self.btn_save_2d = ttk.Button(btn_frame, text="💾 Сохранить 2D", command=self.save_2d)
        self.btn_save_2d.pack(side=tk.LEFT, padx=2, expand=True, fill=tk.X)
        self.btn_save_3d = ttk.Button(btn_frame, text="📊 Сохранить 3D", command=self.save_3d)
        self.btn_save_3d.pack(side=tk.LEFT, padx=2, expand=True, fill=tk.X)
        self.btn_export = ttk.Button(left_panel, text="📁 Экспорт SVG", command=self.export_vector)
        self.btn_export.pack(fill=tk.X, pady=3)
        
        # --- Правая панель: графики ---
                # --- Правая панель: ОДИН БОЛЬШОЙ 2D ГРАФИК НА ВСЮ ОБЛАСТЬ ---
        self.fig = Figure(figsize=(14, 9), dpi=100)
        self.fig.subplots_adjust(left=0.08, right=0.92, bottom=0.08, top=0.92)
        
        self.ax2d = self.fig.add_subplot(111)  # Занимает 100% пространства фигуры
        self.ax2d.set_title("План рельефа с горизонталями", fontsize=14, weight='bold')
        self.ax2d.set_xlabel("X (пиксели)", fontsize=11)
        self.ax2d.set_ylabel("Y (пиксели)", fontsize=11)
        self.ax2d.grid(True, alpha=0.3)
        self.ax2d.set_aspect('equal')
        
        self.canvas = FigureCanvasTkAgg(self.fig, right_panel)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self.toolbar = NavigationToolbar2Tk(self.canvas, right_panel)
        self.toolbar.update()
        
        # Подключаем интерактивный зум колесиком мыши к области графика
        self.canvas.mpl_connect('scroll_event', self.zoom_factory)
        
        # Нижняя панель управления (удален слайдер 3D прозрачности)
                # Нижняя панель управления (старый слайдер удален, параметры управляются из левого меню)
        slider_frame = ttk.Frame(right_panel)
        slider_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(slider_frame, text="ℹ️ Горизонтали рассчитываются автоматически на основе параметров плана", font=('Arial', 10, 'italic'), foreground="gray").pack(side=tk.LEFT, padx=10)
        ttk.Button(slider_frame, text="🔄 Сбросить масштаб зума", command=self.reset_view).pack(side=tk.RIGHT, padx=20)

        self.contour_slider = tk.Scale(slider_frame, from_=5, to=25, orient=tk.HORIZONTAL, resolution=1, command=self.on_contour_change, length=200)
        self.contour_slider.set(10)
        self.contour_slider.pack(side=tk.LEFT, padx=5)
        
        ttk.Button(slider_frame, text="Сбросить масштаб и вид", command=self.reset_view).pack(side=tk.LEFT, padx=20)

    
    def load_image(self):
        """Загрузка изображения без автоматического поиска точек"""
        file_path = filedialog.askopenfilename(
            title="Выберите изображение",
            filetypes=[("Изображения", "*.png *.jpg *.jpeg *.bmp"), ("Все файлы", "*.*")]
        )
        if file_path:
            try:
                pil_img = Image.open(file_path)
                if pil_img.mode in ['RGBA', 'P']:
                    pil_img = pil_img.convert('RGB')
                img = np.array(pil_img)
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                self.original_img = img.copy()
                h, w = img.shape[:2]
                self.img_label.config(text=f"Загружено: {os.path.basename(file_path)} ({w}x{h})")
                
                # Полный сброс старых данных при загрузке новой карты
                self.points_pixels = None
                self.heights = None
                self.is_processed = False
                self.contour_lines = []
                self.points_info_label.config(text="Карта загружена. Нажмите 'НАЧАТЬ РАЗМЕТКУ КАРТЫ'", foreground="blue")
                self.update_preview()
                
                # СТРОКА С АВТОМАТИЧЕСКИМ ПОИСКОМ ОТСЮДА ПОЛНОСТЬЮ УДАЛЕНА
                
            except Exception as e:
                messagebox.showerror("Ошибка", f"Не удалось загрузить: {str(e)}")


    def auto_detect_points(self):
        pass

    def enable_manual_mode(self):
        """Ручной ввод кликами с автоматическим расчетом высоты окна под экран"""
        if self.original_img is None:
            messagebox.showerror("Ошибка", "Сначала загрузите изображение!")
            return
        self.temp_points = []
        self.manual_window = tk.Toplevel(self.root)
        self.manual_window.title("Ручной ввод точек — кликайте по изображению")
        
        img_copy = self.original_img.copy()
        h, w = img_copy.shape[:2]
        
        # Ограничиваем максимальный размер превью, чтобы оно влезало на стандартные дисплеи ноутбуков
        max_size = 650
        scale = min(max_size / w, max_size / h)
        new_w, new_h = int(w * scale), int(h * scale)
        
        # ИСПРАВЛЕНИЕ 3: Выделяем фиксированное место под панель кнопок внизу (+120 пикселей к высоте)
        window_width = max(new_w + 40, 500)
        window_height = new_h + 130
        self.manual_window.geometry(f"{window_width}x{window_height}")
        
        img_scaled = cv2.resize(img_copy, (new_w, new_h))
        self.scale_x = w / new_w
        self.scale_y = h / new_h
        img_rgb = cv2.cvtColor(img_scaled, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img_rgb)
        self.manual_img_tk = ImageTk.PhotoImage(img_pil)
        
        self.manual_canvas = tk.Canvas(self.manual_window, width=new_w, height=new_h, cursor="cross", bg='white')
        self.manual_canvas.pack(pady=10)
        self.manual_canvas.create_image(0, 0, anchor=tk.NW, image=self.manual_img_tk)
        self.manual_canvas.bind("<Button-1>", self.on_manual_click)
        
        info_frame = ttk.Frame(self.manual_window)
        info_frame.pack(pady=5)
        self.point_counter_label = ttk.Label(info_frame, text="Добавлено точек: 0", font=('Arial', 12, 'bold'), foreground="blue")
        self.point_counter_label.pack()
        
        btn_frame = ttk.Frame(self.manual_window)
        btn_frame.pack(pady=5, fill=tk.X)
        
        # Центрируем кнопки на панели
        inner_btn_frame = ttk.Frame(btn_frame)
        inner_btn_frame.pack(anchor=tk.CENTER)
        ttk.Button(inner_btn_frame, text="✅ ЗАКОНЧИТЬ", command=self.save_manual_points).pack(side=tk.LEFT, padx=10)
        ttk.Button(inner_btn_frame, text="🔄 ОЧИСТИТЬ", command=self.clear_temp_points).pack(side=tk.LEFT, padx=10)
        ttk.Button(inner_btn_frame, text="❌ ОТМЕНИТЬ", command=self.cancel_manual_points).pack(side=tk.LEFT, padx=10)


    
    def on_manual_click(self, event):
        x = float(event.x * self.scale_x)
        y = float(event.y * self.scale_y)
        x = max(0, min(x, self.original_img.shape[1] - 1))
        y = max(0, min(y, self.original_img.shape[0] - 1))
        self.temp_points.append([x, y])
        self.point_counter_label.config(text=f"Добавлено точек: {len(self.temp_points)}")
        
        canvas_x = int(event.x)
        canvas_y = int(event.y)
        self.manual_canvas.create_oval(canvas_x-6, canvas_y-6, canvas_x+6, canvas_y+6, fill="red", outline="white", width=2)
        self.manual_canvas.create_text(canvas_x+10, canvas_y-10, text=str(len(self.temp_points)), fill="yellow", font=('Arial', 10, 'bold'))

    def save_manual_points(self):
        if len(self.temp_points) >= 3:
            self.points_pixels = np.array(self.temp_points)
            self.points_info_label.config(text=f"✓ ВСЕГО ТОЧЕК: {len(self.points_pixels)} (вручную)", foreground="green")
            self.update_preview()
            self.manual_window.destroy()
            messagebox.showinfo("Успех", f"Добавлено {len(self.points_pixels)} точек!\nТеперь введите высоты.")
        else:
            messagebox.showerror("Ошибка", f"Нужно минимум 3 точки. Добавлено: {len(self.temp_points)}")

    def cancel_manual_points(self):
        self.temp_points = []
        if hasattr(self, 'manual_window') and self.manual_window:
            self.manual_window.destroy()

    def clear_temp_points(self):
        self.temp_points = []
        self.point_counter_label.config(text="Добавлено точек: 0")
        self.manual_canvas.delete("all")
        self.manual_canvas.create_image(0, 0, anchor=tk.NW, image=self.manual_img_tk)
    def clear_all(self):
        """Полная очистка всех данных"""
        self.points_pixels = None
        self.points_meters = None
        self.heights = None
        self.is_processed = False
        self.contour_lines = []
        self.points_info_label.config(text="")
        self.update_preview()
        
        # Полностью очищаем графики
        self.ax2d.clear()
        self.ax3d.clear()
        self.ax2d.set_title("План рельефа с горизонталями", fontsize=12)
        self.ax2d.set_xlabel("X (м)")
        self.ax2d.set_ylabel("Y (м)")
        self.ax2d.grid(True, alpha=0.3)
        self.ax2d.set_aspect('equal', adjustable='box')
        self.ax3d.set_title("3D поверхность рельефа", fontsize=12)
        
        if hasattr(self, 'cbar') and self.cbar:
            try:
                self.cbar.remove()
            except Exception:
                pass
            self.cbar = None
            
        self.canvas.draw()
        messagebox.showinfo("Очистка", "Все точки удалены.")

    def update_preview(self):
        """Обновление превью в левом меню с приведением координат к int для OpenCV"""
        if self.original_img is None:
            placeholder = np.ones((200, 350, 3), dtype=np.uint8) * 240
            cv2.putText(placeholder, "Изображение не загружено", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 100, 100), 2)
            img_copy = placeholder
        else:
            img_copy = self.original_img.copy()
            if self.points_pixels is not None:
                for i, (x, y) in enumerate(self.points_pixels):
                    # Принудительно округляем координаты до целых для корректной работы cv2.circle
                    ix, iy = int(round(x)), int(round(y))
                    cv2.circle(img_copy, (ix, iy), 8, (0, 0, 255), -1)
                    cv2.circle(img_copy, (ix, iy), 10, (255, 255, 255), 2)
                    cv2.putText(img_copy, str(i+1), (ix+10, iy-8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        h, w = img_copy.shape[:2]
        max_w, max_h = 350, 250
        scale = min(max_w / w, max_h / h, 1.0)
        if scale < 1.0:
            new_w, new_h = int(w * scale), int(h * scale)
            img_copy = cv2.resize(img_copy, (new_w, new_h))
        img_rgb = cv2.cvtColor(img_copy, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img_rgb)
        img_tk = ImageTk.PhotoImage(img_pil)
        self.img_preview.config(image=img_tk)
        self.img_preview.image = img_tk


    def input_heights(self):
        """Ввод высот вручную"""
        if self.points_pixels is None:
            messagebox.showerror("Ошибка", "Сначала найдите или введите точки!")
            return
        dialog = tk.Toplevel(self.root)
        dialog.title("Ввод высот точек")
        dialog.geometry("500x500")
        ttk.Label(dialog, text=f"Введите высоты для {len(self.points_pixels)} точек:", font=('Arial', 10, 'bold')).pack(pady=10)
        
        canvas_frame = ttk.Frame(dialog)
        canvas_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        canvas = tk.Canvas(canvas_frame)
        scrollbar = ttk.Scrollbar(canvas_frame, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)
        scrollable_frame.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        height_entries = []
        for i, (x, y) in enumerate(self.points_pixels):
            frame = ttk.Frame(scrollable_frame)
            frame.pack(fill=tk.X, pady=3)
            ttk.Label(frame, text=f"Точка {i+1}:", width=8).pack(side=tk.LEFT, padx=5)
            ttk.Label(frame, text=f"({x}, {y})", width=12).pack(side=tk.LEFT, padx=5)
            ttk.Label(frame, text="Высота (м):").pack(side=tk.LEFT, padx=5)
            entry = ttk.Entry(frame, width=12)
            entry.pack(side=tk.LEFT, padx=5)
            if self.heights is not None and i < len(self.heights):
                entry.insert(0, str(self.heights[i]))
            height_entries.append(entry)
            
        def save():
            heights = []
            for entry in height_entries:
                try:
                    h = float(entry.get()) if entry.get() else 0.0
                    heights.append(h)
                except ValueError:
                    heights.append(0.0)
            self.heights = np.array(heights)
            dialog.destroy()
            messagebox.showinfo("Успех", "Высоты сохранены!\nТеперь нажмите ПОСТРОИТЬ РЕЛЬЕФ")
            
        ttk.Button(dialog, text="Сохранить", command=save).pack(pady=10)

    def load_heights_from_csv(self):
        """Загрузка высот из CSV/TXT файла"""
        if self.points_pixels is None:
            messagebox.showerror("Ошибка", "Сначала найдите или введите точки!")
            return
        file_path = filedialog.askopenfilename(
            title="Выберите файл с высотами",
            filetypes=[("CSV/TXT файлы", "*.csv *.txt"), ("Все файлы", "*.*")]
        )
        if file_path:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    lines = f.readlines()
                heights = []
                for line in lines:
                    line = line.strip()
                    if line and not line.startswith('#'):
                        parts = line.replace(',', ' ').split()
                        for part in parts:
                            try:
                                h = float(part)
                                heights.append(h)
                                break
                            except ValueError:
                                continue
                if len(heights) == len(self.points_pixels):
                    self.heights = np.array(heights)
                    messagebox.showinfo("УSuccess", f"Загружено {len(heights)} высот!\nТеперь нажмите ПОСТРОИТЬ РЕЛЬЕФ")
                else:
                    messagebox.showerror("Ошибка", f"Количество высот ({len(heights)}) не совпадает с количеством точек ({len(self.points_pixels)})")
            except Exception as e:
                messagebox.showerror("Ошибка", f"Не удалось прочитать файл: {e}")

    
    def process_data(self):
        """Построение рельефа с автоматическим расчетом горизонталей и защитой от отрицательного ввода"""
        if self.original_img is None:
            messagebox.showerror("Ошибка данных", "Сначала загрузите изображение плана/карты!")
            return

        if self.points_pixels is None or len(self.points_pixels) == 0:
            messagebox.showerror(
                "Ошибка данных", 
                "Точки рельефа не найдены!\n\n"
                "Пожалуйста, сначала нажмите синюю кнопку \n'🎯 НАЧАТЬ РАЗМЕТКУ КАРТЫ'\n"
                "и отметьте кликами мыши центры кружков."
            )
            return
            
        if self.heights is None or len(self.heights) != len(self.points_pixels):
            messagebox.showerror("Ошибка соответствия", "Количество введенных высот не совпадает с количеством точек!")
            return
            
        try:
            self.btn_process.config(text="⏳ ПОСТРОЕНИЕ...", state=tk.DISABLED)
            self.root.update()
            
            # 1. Считывание параметров и базовая валидация типов данных
            try:
                scale_val = float(self.scale_entry.get())
                step_mm_val = float(self.step_mm_entry.get())
            except ValueError:
                messagebox.showerror("Ошибка параметров", "Убедитесь, что масштаб и шаг палетки введены числами!")
                self.btn_process.config(text="▶ ПОСТРОИТЬ РЕЛЬЕФ", state=tk.NORMAL)
                return

            # 2. ОБРАБОТКА ОТРИЦАТЕЛЬНОГО И НУЛЕВОГО ВВОДА
            if scale_val <= 0 or step_mm_val <= 0:
                messagebox.showerror(
                    "Некорректные параметры", 
                    "Масштаб плана и шаг палетки должны быть строго больше нуля!\n\n"
                    "Отрицательные значения или ноль недопустимы в топографии.\n"
                    "Параметры сброшены на значения по умолчанию."
                )
                # Автоматический возврат безопасных дефолтных значений в интерфейс
                self.scale_entry.delete(0, tk.END)
                self.scale_entry.insert(0, "1000")
                self.step_mm_entry.delete(0, tk.END)
                self.step_mm_entry.insert(0, "10")
                
                self.btn_process.config(text="▶ ПОСТРОИТЬ РЕЛЬЕФ", state=tk.NORMAL)
                return

            # Если проверка пройдена, фиксируем параметры
            self.scale = scale_val
            self.palette_step_mm = step_mm_val
            
            self.points_pixels = np.array(self.points_pixels, dtype=float)
            h_img, w_img = self.original_img.shape[:2]
            
            # Генерируем плотную сетку 300х300 строго в границах изображения
            x_coords = np.linspace(0, w_img, 300)
            y_coords = np.linspace(0, h_img, 300)
            self.grid_x, self.grid_y = np.meshgrid(x_coords, y_coords)
            
            # Сплайн-интерполяция методом RBF
            from scipy.interpolate import RBFInterpolator
            grid_flat = np.vstack([self.grid_x.ravel(), self.grid_y.ravel()]).T
            rbf = RBFInterpolator(self.points_pixels, self.heights, kernel='cubic', smoothing=1.0)
            
            grid_heights_flat = rbf(grid_flat)
            self.grid_heights = grid_heights_flat.reshape(self.grid_x.shape)
            
            # Гауссово сглаживание изолиний
            from scipy.ndimage import gaussian_filter
            self.grid_heights = gaussian_filter(self.grid_heights, sigma=1.5)
            
            # Автоматический расчет высоты сечения рельефа: h = (шаг_мм * масштаб) / 1000
            height_interval = (self.palette_step_mm * self.scale) / 1000.0
            
            h_min, h_max = self.heights.min(), self.heights.max()
            start_level = np.ceil(h_min / height_interval) * height_interval
            end_level = np.floor(h_max / height_interval) * height_interval
            
            if start_level >= end_level:
                self.contour_levels = np.linspace(h_min, h_max, 5)
            else:
                self.contour_levels = np.arange(start_level, end_level + height_interval, height_interval)
                if len(self.contour_levels) == 0:
                    self.contour_levels = np.linspace(h_min, h_max, 10)
            
            self.is_processed = True
            self._initial_limits_set = False 
            
            if hasattr(self, 'cbar') and self.cbar:
                try: self.cbar.remove()
                except Exception: pass
                self.cbar = None
                
            self.full_redraw()
            
            self.btn_process.config(text="▶ ПОСТРОИТЬ РЕЛЬЕФ", state=tk.NORMAL)
            messagebox.showinfo("Успех", f"Рельеф построен!\nШаг горизонталей: {height_interval:.2f} м (вычислен автоматически).")
        except Exception as e:
            self.btn_process.config(text="▶ ПОСТРОИТЬ РЕЛЬЕФ", state=tk.NORMAL)
            messagebox.showerror("Ошибка расчета", f"Ошибка во время интерполяции: {str(e)}")

    

    def redraw_2d_only(self):
        """Перерисовка только 2D графика для быстродействия переключателей"""
        if not self.is_processed:
            return
        self.full_redraw()

    def zoom_factory(self, event):
        """Плавное интерактивное масштабирование (зум) плана колесиком мыши"""
        if event.inaxes != self.ax2d:
            return  # Зум работает только если курсор находится над полем графика
            
        # Коэффициент масштабирования за один щелчок колесика
        base_scale = 1.25
        
        # Получаем текущие границы осей
        cur_xlim = self.ax2d.get_xlim()
        cur_ylim = self.ax2d.get_ylim()
        
        xdata = event.xdata  # Координата X курсора мыши
        ydata = event.ydata  # Координата Y курсора мыши
        
        if event.button == 'up':
            # Увеличение масштаба (Приближение)
            scale_factor = 1 / base_scale
        elif event.button == 'down':
            # Уменьшение масштаба (Отдаление)
            scale_factor = base_scale
        else:
            return
            
        # Вычисляем новые границы относительно положения курсора мыши
        new_xmin = xdata - (xdata - cur_xlim[0]) * scale_factor
        new_xmax = xdata + (cur_xlim[1] - xdata) * scale_factor
        new_ymin = ydata - (ydata - cur_ylim[0]) * scale_factor
        new_ymax = ydata + (cur_ylim[1] - ydata) * scale_factor
        
        # Задаем новые границы осей
        self.ax2d.set_xlim([new_xmin, new_xmax])
        self.ax2d.set_ylim([new_ymin, new_ymax])
        
        # Перерисовываем только холст (без полного пересчета сплайнов) для плавности
        self.canvas.draw()

    def on_contour_change(self, value):
        """Изменение количества горизонталей слайдером"""
        if not self.is_processed:
            return
        new_val = int(float(value))
        if new_val != self.num_contours:
            self.num_contours = new_val
            self.full_redraw()

    def on_alpha_change(self, value):
        pass

    def reset_view(self):
        """Сброс интерактивного зума и масштаба осей к исходным размерам картинки"""
        self.num_contours = 10
        self.contour_slider.set(10)
        self._initial_limits_set = False
        
        if self.is_processed:
            self.full_redraw()
        else:
            self.ax2d.clear()
            self.ax2d.set_title("План рельефа с горизонталями", fontsize=12)
            self.ax2d.set_aspect('equal', adjustable='datalim')
            if hasattr(self, 'cbar') and self.cbar:
                try: self.cbar.remove()
                except Exception: pass
                self.cbar = None
            self.canvas.draw()
        messagebox.showinfo("Сброс", "Масштаб плана и положение осей возвращены по умолчанию.")

    def save_3d(self):
        messagebox.showinfo("Инфо", "3D режим отключен в настройках интерфейса.")
    def redraw_2d_only(self):
        """Перерисовка только 2D графика для быстродействия переключателей"""
        if not self.is_processed:
            return
        self.full_redraw()

    def on_contour_change(self, value):
        pass


    def on_alpha_change(self, value):
        """Изменение прозрачности 3D поверхности"""
        if not self.is_processed:
            return
        self.transparency_3d = float(value)
        self.full_redraw()

    def full_redraw(self):
        """Полная перерисовка плоского 2D плана со светло-серой цельной сеткой поверх карты"""
        if not self.is_processed or self.original_img is None:
            return
            
        h_img, w_img = self.original_img.shape[:2]
        
        # Считываем текущий масштаб зума пользователя
        xlim = self.ax2d.get_xlim()
        ylim = self.ax2d.get_ylim()
        has_limits = hasattr(self, '_initial_limits_set') and self._initial_limits_set
        
        self.ax2d.clear()
        
        self.ax2d.set_title("План рельефа с горизонталями (контроль точек)", fontsize=13, weight='bold')
        self.ax2d.set_xlabel("X (пиксели)", fontsize=10)
        self.ax2d.set_ylabel("Y (пиксели)", fontsize=10)
        
        # Вывод фоновой карты
        try:
            img_rgb = cv2.cvtColor(self.original_img, cv2.COLOR_BGR2RGB)
            self.ax2d.imshow(img_rgb, origin='upper', zorder=1) 
        except Exception:
            pass
            
        # Жесткая фиксация осей
        if not has_limits:
            self.ax2d.set_xlim([0, w_img])
            self.ax2d.set_ylim([h_img, 0])
            self._initial_limits_set = True
            xlim = [0, w_img]
            ylim = [h_img, 0]
        
        if self.heights is None or self.grid_x is None or self.grid_y is None or self.grid_heights is None:
            return
            
        # ИСПРАВЛЕНИЕ: Отрисовка цельной светло-серой сетки поверх карты
        if self.show_grid_var.get():
            # grid(True) включает сетку, linestyle='-' делает её цельной
            # color='lightgray' задает светло-серый цвет, zorder=3 выводит её поверх картинки
            self.ax2d.grid(True, which='both', color='#D3D3D3', linestyle='-', linewidth=0.8, alpha=0.7, zorder=3)
        else:
            self.ax2d.grid(False)
            
        # Отрисовка горизонталей рельефа (zorder=4, поверх сетки)
        contours = self.ax2d.contour(self.grid_x, self.grid_y, self.grid_heights, levels=self.contour_levels, colors='cyan', linewidths=1.8, zorder=4)
        self.ax2d.clabel(contours, inline=True, fmt='%.1f', fontsize=9, colors='white', inline_spacing=3)
        
        # Отрисовка маркеров точек (zorder=10)
        if self.show_points_var.get() and self.points_pixels is not None and len(self.points_pixels) > 0:
            scatter = self.ax2d.scatter(self.points_pixels[:, 0], self.points_pixels[:, 1], c='red', s=100, edgecolors='white', linewidth=1.5, zorder=10)
            
            for i, (x, y) in enumerate(self.points_pixels):
                if i < len(self.heights):
                    self.ax2d.annotate(
                        f"{i+1}({self.heights[i]}м)", 
                        (x, y), 
                        xytext=(8, -8), 
                        textcoords='offset points', 
                        fontsize=9, 
                        color='black',
                        weight='bold',
                        bbox=dict(boxstyle='round,pad=0.2', facecolor='yellow', alpha=0.8),
                        zorder=11
                    )
                    
            if not hasattr(self, 'cbar') or self.cbar is None:
                self.cbar = self.fig.colorbar(scatter, ax=self.ax2d, label='Высота (м)', shrink=0.8)
            else:
                try: self.cbar.update_normal(scatter)
                except Exception: pass
                    
        # Возвращаем текущий масштаб зума
        self.ax2d.set_xlim(xlim)
        self.ax2d.set_ylim(ylim)
        self.ax2d.set_aspect('equal', adjustable='box')
        
        self.canvas.draw()


    def reset_view(self):
        """Сброс всех интерактивных параметров и графиков"""
        self.num_contours = 10
        self.transparency_3d = 0.7
        self.contour_slider.set(10)
        self.alpha_slider.set(0.7)
        
        if self.is_processed:
            # Возврат камеры в дефолтное положение
            self.ax3d.view_init(elev=30, azim=-60)
            self.full_redraw()
        else:
            self.ax2d.clear()
            self.ax3d.clear()
            self.ax2d.set_title("План рельефа с горизонталями", fontsize=12)
            self.ax2d.set_aspect('equal', adjustable='box')
            self.ax3d.set_title("3D поверхность рельефа", fontsize=12)
            if hasattr(self, 'cbar') and self.cbar:
                try:
                    self.cbar.remove()
                except Exception:
                    pass
                self.cbar = None
            self.canvas.draw()
        messagebox.showinfo("Сброс", "Параметры отображения возвращены по умолчанию.")

    def save_2d(self):
        if not self.is_processed:
            messagebox.showerror("Ошибка", "Сначала постройте рельеф!")
            return
        path = filedialog.asksaveasfilename(defaultextension=".png", filetypes=[("PNG", "*.png")])
        if path:
            # Экспортируем только левую 2D половину фигуры
            extent = self.ax2d.get_window_extent().transformed(self.fig.dpi_scale_trans.inverted())
            self.fig.savefig(path, dpi=300, bbox_inches=extent.expanded(1.3, 1.3))
            messagebox.showinfo("Успех", f"2D сохранен: {path}")

    def save_3d(self):
        if not self.is_processed:
            messagebox.showerror("Ошибка", "Сначала постройте рельеф!")
            return
        path = filedialog.asksaveasfilename(defaultextension=".png", filetypes=[("PNG", "*.png")])
        if path:
            # Экспортируем только правую 3D половину фигуры
            extent = self.ax3d.get_window_extent().transformed(self.fig.dpi_scale_trans.inverted())
            self.fig.savefig(path, dpi=300, bbox_inches=extent.expanded(1.3, 1.3))
            messagebox.showinfo("Успех", f"3D сохранен: {path}")

    def export_vector(self):
        if not self.is_processed:
            messagebox.showerror("Ошибка", "Сначала постройте рельеф!")
            return
        path = filedialog.asksaveasfilename(defaultextension=".svg", filetypes=[("SVG", "*.svg")])
        if path:
            extent = self.ax2d.get_window_extent().transformed(self.fig.dpi_scale_trans.inverted())
            self.fig.savefig(path, format='svg', bbox_inches=extent.expanded(1.3, 1.3))
            messagebox.showinfo("Успех", f"План экспортирован в SVG: {path}")
    
def main():
    root = tk.Tk()
    app = ReliefInterpolationApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()



