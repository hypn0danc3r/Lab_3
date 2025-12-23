"""
Лабораторная работа 3, Вариант 5
Обработка изображений: построение и эквализация гистограммы + линейное контрастирование
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk, ImageStat
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import os


class ImageProcessorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Лабораторная работа 3 - Вариант 5: Обработка изображений")
        self.root.geometry("1200x800")
        
        self.original_image = None
        self.processed_image = None
        self.image_path = None
        
        self.create_widgets()
        
    def create_widgets(self):
        # Главный контейнер
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Заголовок
        title_label = ttk.Label(main_frame, text="Обработка изображений: Гистограмма и контрастирование", 
                               font=("Arial", 14, "bold"))
        title_label.pack(pady=(0, 10))
        
        # Верхняя панель с кнопками
        control_frame = ttk.Frame(main_frame)
        control_frame.pack(fill=tk.X, pady=5)
        
        ttk.Button(control_frame, text="Загрузить изображение", 
                  command=self.load_image).pack(side=tk.LEFT, padx=5)
        ttk.Button(control_frame, text="Построить гистограмму", 
                  command=self.show_histogram).pack(side=tk.LEFT, padx=5)
        ttk.Button(control_frame, text="Линейное контрастирование", 
                  command=self.linear_contrast).pack(side=tk.LEFT, padx=5)
        ttk.Button(control_frame, text="Эквализация гистограммы", 
                  command=self.equalize_histogram).pack(side=tk.LEFT, padx=5)
        ttk.Button(control_frame, text="Сравнить методы", 
                  command=self.compare_methods).pack(side=tk.LEFT, padx=5)
        ttk.Button(control_frame, text="Сохранить результат", 
                  command=self.save_image).pack(side=tk.LEFT, padx=5)
        
        # Параметры линейного контрастирования
        params_frame = ttk.LabelFrame(main_frame, text="Параметры линейного контрастирования", padding="5")
        params_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(params_frame, text="Минимум:").pack(side=tk.LEFT, padx=5)
        self.min_val = tk.IntVar(value=0)
        min_spin = ttk.Spinbox(params_frame, from_=0, to=255, textvariable=self.min_val, width=5)
        min_spin.pack(side=tk.LEFT, padx=5)
        
        ttk.Label(params_frame, text="Максимум:").pack(side=tk.LEFT, padx=5)
        self.max_val = tk.IntVar(value=255)
        max_spin = ttk.Spinbox(params_frame, from_=0, to=255, textvariable=self.max_val, width=5)
        max_spin.pack(side=tk.LEFT, padx=5)
        
        ttk.Button(params_frame, text="Автоматический расчет", 
                  command=self.auto_contrast_params).pack(side=tk.LEFT, padx=10)
        
        # Область для изображений
        images_frame = ttk.Frame(main_frame)
        images_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        
        # Оригинальное изображение
        original_frame = ttk.LabelFrame(images_frame, text="Оригинальное изображение", padding="5")
        original_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5)
        
        self.original_label = ttk.Label(original_frame, text="Загрузите изображение", 
                                       background="lightgray", anchor=tk.CENTER)
        self.original_label.pack(fill=tk.BOTH, expand=True)
        
        # Обработанное изображение
        processed_frame = ttk.LabelFrame(images_frame, text="Обработанное изображение", padding="5")
        processed_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5)
        
        self.processed_label = ttk.Label(processed_frame, text="Результат обработки", 
                                         background="lightgray", anchor=tk.CENTER)
        self.processed_label.pack(fill=tk.BOTH, expand=True)
        
        # Статус бар
        self.status_var = tk.StringVar()
        self.status_var.set("Готов к работе. Загрузите изображение.")
        status_bar = ttk.Label(main_frame, textvariable=self.status_var, 
                             relief=tk.SUNKEN, anchor=tk.W)
        status_bar.pack(fill=tk.X, pady=(5, 0))
    
    def load_image(self):
        """Загрузить изображение"""
        file_path = filedialog.askopenfilename(
            title="Выберите изображение",
            filetypes=[("Изображения", "*.jpg *.jpeg *.png *.bmp *.gif *.tiff"), 
                      ("Все файлы", "*.*")]
        )
        
        if file_path:
            try:
                self.image_path = file_path
                self.original_image = Image.open(file_path)
                
                # Конвертируем в RGB если нужно
                if self.original_image.mode != 'RGB':
                    self.original_image = self.original_image.convert('RGB')
                
                # Отображаем оригинальное изображение
                self.display_image(self.original_image, self.original_label)
                
                # Сбрасываем обработанное изображение
                self.processed_image = None
                self.processed_label.config(image='', text="Результат обработки")
                
                self.status_var.set(f"Изображение загружено: {os.path.basename(file_path)}")
            except Exception as e:
                messagebox.showerror("Ошибка", f"Не удалось загрузить изображение:\n{str(e)}")
    
    def display_image(self, image, label_widget, max_size=(500, 500)):
        """Отобразить изображение в виджете"""
        # Изменяем размер для отображения
        img_copy = image.copy()
        img_copy.thumbnail(max_size, Image.Resampling.LANCZOS)
        
        # Конвертируем в PhotoImage
        photo = ImageTk.PhotoImage(img_copy)
        label_widget.config(image=photo, text='')
        label_widget.image = photo  # Сохраняем ссылку
    
    def show_histogram(self):
        """Построить гистограмму изображения"""
        if self.original_image is None:
            messagebox.showwarning("Предупреждение", "Сначала загрузите изображение!")
            return
        
        # Получаем гистограмму
        histogram = self.original_image.histogram()
        
        # Разделяем на каналы RGB
        r_hist = histogram[0:256]
        g_hist = histogram[256:512]
        b_hist = histogram[512:768]
        
        # Создаем окно для гистограммы
        hist_window = tk.Toplevel(self.root)
        hist_window.title("Гистограмма изображения")
        hist_window.geometry("900x700")
        
        # Фрейм для управления
        control_frame = ttk.Frame(hist_window, padding="10")
        control_frame.pack(fill=tk.X)
        
        # Чекбоксы для выбора каналов
        channels_frame = ttk.LabelFrame(control_frame, text="Выберите каналы для отображения", padding="5")
        channels_frame.pack(side=tk.LEFT, padx=5)
        
        show_red = tk.BooleanVar(value=True)
        show_green = tk.BooleanVar(value=True)
        show_blue = tk.BooleanVar(value=True)
        
        # Создаем фигуру и ось
        fig, ax = plt.subplots(figsize=(9, 6))
        canvas = FigureCanvasTkAgg(fig, hist_window)
        
        # Функция обновления графика
        def update_histogram():
            ax.clear()
            x = range(256)
            has_any = False
            
            if show_red.get():
                ax.plot(x, r_hist, color='red', linewidth=2, alpha=0.8, label='Red')
                has_any = True
            if show_green.get():
                ax.plot(x, g_hist, color='green', linewidth=2, alpha=0.8, label='Green')
                has_any = True
            if show_blue.get():
                ax.plot(x, b_hist, color='blue', linewidth=2, alpha=0.8, label='Blue')
                has_any = True
            
            if not has_any:
                ax.text(0.5, 0.5, 'Выберите хотя бы один канал', 
                       transform=ax.transAxes, ha='center', va='center',
                       fontsize=14, color='gray')
            else:
                ax.set_xlabel('Интенсивность (0-255)')
                ax.set_ylabel('Количество пикселей')
                ax.set_title('Гистограмма изображения')
                ax.legend()
                ax.grid(True, alpha=0.3)
            
            canvas.draw()
        
        # Создаем чекбоксы с привязкой к функции обновления
        ttk.Checkbutton(channels_frame, text="Red (Красный)", variable=show_red,
                       command=update_histogram).pack(side=tk.LEFT, padx=5)
        ttk.Checkbutton(channels_frame, text="Green (Зеленый)", variable=show_green,
                       command=update_histogram).pack(side=tk.LEFT, padx=5)
        ttk.Checkbutton(channels_frame, text="Blue (Синий)", variable=show_blue,
                       command=update_histogram).pack(side=tk.LEFT, padx=5)
        
        # Первоначальное построение
        update_histogram()
        
        # Размещаем canvas
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        self.status_var.set("Гистограмма построена. Используйте чекбоксы для выбора каналов.")
    
    def linear_contrast(self):
        """Применить линейное контрастирование"""
        if self.original_image is None:
            messagebox.showwarning("Предупреждение", "Сначала загрузите изображение!")
            return
        
        min_val = self.min_val.get()
        max_val = self.max_val.get()
        
        if min_val >= max_val:
            messagebox.showerror("Ошибка", "Минимум должен быть меньше максимума!")
            return
        
        # Конвертируем изображение в массив
        img_array = np.array(self.original_image, dtype=np.float64)
        
        # Применяем линейное контрастирование для каждого канала отдельно
        result_channels = []
        for i in range(3):  # RGB каналы
            channel = img_array[:, :, i]
            
            # Находим текущие минимум и максимум для этого канала
            current_min = channel.min()
            current_max = channel.max()
            
            # Применяем линейное контрастирование
            # Формула: new_value = (value - current_min) * (max_val - min_val) / (current_max - current_min) + min_val
            if current_max != current_min:
                channel = (channel - current_min) * (max_val - min_val) / (current_max - current_min) + min_val
                channel = np.clip(channel, 0, 255)
            else:
                # Если все пиксели одинаковые, устанавливаем среднее значение
                channel = np.full_like(channel, (min_val + max_val) / 2)
            
            result_channels.append(channel)
        
        # Объединяем каналы обратно
        img_array = np.dstack(result_channels).astype(np.uint8)
        
        # Создаем новое изображение
        self.processed_image = Image.fromarray(img_array)
        
        # Отображаем результат
        self.display_image(self.processed_image, self.processed_label)
        
        self.status_var.set(f"Линейное контрастирование применено (диапазон: {min_val}-{max_val})")
    
    def auto_contrast_params(self):
        """Автоматически рассчитать параметры контрастирования (максимальный эффект)"""
        if self.original_image is None:
            messagebox.showwarning("Предупреждение", "Сначала загрузите изображение!")
            return
        
        # Конвертируем в массив для анализа
        img_array = np.array(self.original_image)
        
        # Находим реальный диапазон для каждого канала
        r_min, r_max = img_array[:, :, 0].min(), img_array[:, :, 0].max()
        g_min, g_max = img_array[:, :, 1].min(), img_array[:, :, 1].max()
        b_min, b_max = img_array[:, :, 2].min(), img_array[:, :, 2].max()
        
        # Берем общий минимум и максимум по всем каналам
        overall_min = min(r_min, g_min, b_min)
        overall_max = max(r_max, g_max, b_max)
        
        # Всегда устанавливаем максимальный диапазон 0-255 для максимального эффекта контрастирования
        # Это растянет текущий диапазон изображения на весь доступный диапазон
        min_param = 0
        max_param = 255
        
        self.min_val.set(min_param)
        self.max_val.set(max_param)
        
        self.status_var.set(f"Параметры установлены: 0-255 (текущий диапазон изображения: {overall_min}-{overall_max})")
    
    def equalize_histogram(self):
        """Применить эквализацию гистограммы"""
        if self.original_image is None:
            messagebox.showwarning("Предупреждение", "Сначала загрузите изображение!")
            return
        
        # Конвертируем в массив
        img_array = np.array(self.original_image)
        
        # Применяем эквализацию для каждого канала отдельно
        equalized_channels = []
        for i in range(3):  # RGB каналы
            channel = img_array[:, :, i]
            # Вычисляем кумулятивную функцию распределения
            hist, bins = np.histogram(channel.flatten(), 256, [0, 256])
            cdf = hist.cumsum()
            
            # Нормализуем CDF, избегая деления на ноль
            if cdf.max() != cdf.min():
                cdf_normalized = (cdf - cdf.min()) * 255 / (cdf.max() - cdf.min())
            else:
                # Если все пиксели одинаковые, оставляем как есть
                cdf_normalized = cdf
            
            # Применяем преобразование
            equalized_channel = np.interp(channel.flatten(), bins[:-1], cdf_normalized)
            equalized_channel = equalized_channel.reshape(channel.shape).astype(np.uint8)
            equalized_channels.append(equalized_channel)
        
        # Объединяем каналы
        self.processed_image = Image.fromarray(np.dstack(equalized_channels))
        
        # Отображаем результат
        self.display_image(self.processed_image, self.processed_label)
        
        self.status_var.set("Эквализация гистограммы применена")
    
    def compare_methods(self):
        """Сравнить методы обработки"""
        if self.original_image is None:
            messagebox.showwarning("Предупреждение", "Сначала загрузите изображение!")
            return
        
        # Создаем окно сравнения
        compare_window = tk.Toplevel(self.root)
        compare_window.title("Сравнение методов обработки")
        compare_window.geometry("1400x800")
        
        # Применяем оба метода используя ТОТ ЖЕ алгоритм, что и в основных функциях
        img_array = np.array(self.original_image, dtype=np.float64)
        min_val = self.min_val.get()
        max_val = self.max_val.get()
        
        # Линейное контрастирование (тот же алгоритм, что в linear_contrast)
        result_channels_linear = []
        for i in range(3):  # RGB каналы
            channel = img_array[:, :, i]
            current_min = channel.min()
            current_max = channel.max()
            
            if current_max != current_min:
                channel = (channel - current_min) * (max_val - min_val) / (current_max - current_min) + min_val
                channel = np.clip(channel, 0, 255)
            else:
                channel = np.full_like(channel, (min_val + max_val) / 2)
            
            result_channels_linear.append(channel)
        
        linear_img = np.dstack(result_channels_linear).astype(np.uint8)
        linear_result = Image.fromarray(linear_img)
        
        # Эквализация (тот же алгоритм, что в equalize_histogram)
        equalized_channels = []
        for i in range(3):
            channel = img_array[:, :, i].astype(np.uint8)
            hist, bins = np.histogram(channel.flatten(), 256, [0, 256])
            cdf = hist.cumsum()
            if cdf.max() != cdf.min():
                cdf_normalized = (cdf - cdf.min()) * 255 / (cdf.max() - cdf.min())
            else:
                cdf_normalized = cdf
            equalized_channel = np.interp(channel.flatten(), bins[:-1], cdf_normalized)
            equalized_channel = equalized_channel.reshape(channel.shape).astype(np.uint8)
            equalized_channels.append(equalized_channel)
        equalized_result = Image.fromarray(np.dstack(equalized_channels))
        
        # Создаем фигуру с 4 изображениями
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Конвертируем PIL изображения в numpy массивы для matplotlib
        orig_array = np.array(self.original_image)
        linear_array = np.array(linear_result)
        equal_array = np.array(equalized_result)
        
        axes[0, 0].imshow(orig_array)
        axes[0, 0].set_title('Оригинальное изображение')
        axes[0, 0].axis('off')
        
        axes[0, 1].imshow(linear_array)
        axes[0, 1].set_title('Линейное контрастирование')
        axes[0, 1].axis('off')
        
        axes[1, 0].imshow(equal_array)
        axes[1, 0].set_title('Эквализация гистограммы')
        axes[1, 0].axis('off')
        
        # Гистограммы - показываем все три канала для лучшего понимания
        orig_hist = self.original_image.histogram()
        linear_hist = linear_result.histogram()
        equal_hist = equalized_result.histogram()
        
        x = range(256)
        # Показываем все три канала оригинального изображения (средняя яркость)
        orig_combined = [(orig_hist[i] + orig_hist[i+256] + orig_hist[i+512]) / 3 for i in range(256)]
        linear_combined = [(linear_hist[i] + linear_hist[i+256] + linear_hist[i+512]) / 3 for i in range(256)]
        equal_combined = [(equal_hist[i] + equal_hist[i+256] + equal_hist[i+512]) / 3 for i in range(256)]
        
        axes[1, 1].plot(x, orig_combined, 'r-', linewidth=2, alpha=0.7, label='Оригинал')
        axes[1, 1].plot(x, linear_combined, 'g-', linewidth=2, alpha=0.7, label='Линейное')
        axes[1, 1].plot(x, equal_combined, 'b-', linewidth=2, alpha=0.7, label='Эквализация')
        axes[1, 1].set_title('Сравнение гистограмм (средняя яркость всех каналов)')
        axes[1, 1].set_xlabel('Интенсивность (0-255)')
        axes[1, 1].set_ylabel('Количество пикселей')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        canvas = FigureCanvasTkAgg(fig, compare_window)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        self.status_var.set("Сравнение методов выполнено")
    
    def save_image(self):
        """Сохранить обработанное изображение"""
        if self.processed_image is None:
            messagebox.showwarning("Предупреждение", "Нет обработанного изображения для сохранения!")
            return
        
        file_path = filedialog.asksaveasfilename(
            title="Сохранить изображение",
            defaultextension=".png",
            filetypes=[("PNG", "*.png"), ("JPEG", "*.jpg"), ("Все файлы", "*.*")]
        )
        
        if file_path:
            try:
                self.processed_image.save(file_path)
                self.status_var.set(f"Изображение сохранено: {os.path.basename(file_path)}")
                messagebox.showinfo("Успех", "Изображение успешно сохранено!")
            except Exception as e:
                messagebox.showerror("Ошибка", f"Не удалось сохранить изображение:\n{str(e)}")


def main():
    root = tk.Tk()
    app = ImageProcessorApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
