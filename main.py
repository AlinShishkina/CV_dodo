
import cv2
import pandas as pd
import numpy as np
import argparse
import sys
import os
from datetime import datetime
import warnings
warnings.filterwarnings("ignore")

# Глобальная проверка доступности YOLOv8
YOLO_AVAILABLE = False
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
    print("YOLOv8 доступен")
except ImportError:
    print("Работает только детекция движения (без YOLO)")

def ensure_video(video_path):
    """Проверяет существование входного видеофайла"""
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Видео не найдено: {video_path}")

def get_video_info(video_path):
    """
    Получает метаданные видео: FPS, размеры, длительность
    Возвращает кортеж: (fps, width, height, total_frames, duration)
    """
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    duration = total_frames / fps if fps > 0 else 0
    return fps, width, height, total_frames, duration

class TableDetector:
    def __init__(self, video_path, use_yolo=True):
        """
        Инициализация детектора столиков
        
        Args:
            video_path (str): Путь к видеофайлу
            use_yolo (bool): Использовать YOLOv8 для детекции людей
        """
        self.video_path = video_path
        self.use_yolo = use_yolo and YOLO_AVAILABLE
        self.cap = None
        
        # Загрузка YOLO модели (если доступна)
        if self.use_yolo:
            self.model = YOLO('yolov8n.pt')
            print("YOLOv8n модель загружена")
        
        # Background Subtractor для детекции движения
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=300,      # Количество кадров для обучения фона
            varThreshold=40,  # Порог вариации пикселей
            detectShadows=True # Детекция теней
        )
        print("BackgroundSubtractor инициализирован")
        
        # Основные параметры
        self.table_roi = None  # ROI в формате (x1, y1, width, height)
        self.output_path = "output.mp4"
        self.problem_frames_dir = "problem_frames"
        os.makedirs(self.problem_frames_dir, exist_ok=True)
        self.problem_frames_saved = []
        
        # Состояние детектора
        self.state = "empty"  # Текущее состояние: "empty" или "occupied"
        self.events = []      # Лог событий
        self.empty_frames_count = 0
        self.occupied_frames_count = 0
        
        # Настраиваемые пороги
        self.MOTION_THRESHOLD = 800           # Пикселей движения для срабатывания
        self.MOTION_CONFIRM_FRAMES = 15       # Кадров подтверждения движения
        self.EMPTY_CONFIRM_FRAMES = 60        # Кадров подтверждения пустоты
        self.NEAR_PERSON_DISTANCE = 100       # Расстояние до человека (пиксели)
        self.MAX_PROBLEM_FRAMES = 10          # Максимум проблемных кадров

    def select_roi(self):
        """
        Интерактивный выбор области интереса (ROI) - столика
        Формат ROI: (x1, y1, width, height)
        Управление: ЛКМ-выделить, SPACE-подтвердить, R-сброс, ESC-выход
        """
        print("\nВыбор столика")
        print("ЛКМ: выделить область | SPACE: подтвердить | R: сброс | ESC: выход")
        
        self.cap = cv2.VideoCapture(self.video_path)
        ret, frame = self.cap.read()
        if not ret:
            raise ValueError("Не удалось прочитать первый кадр")
        
        frame_copy = frame.copy()
        drawing = False
        ix, iy = -1, -1
        roi = None
        
        def mouse_callback(event, x, y, flags, param):
            nonlocal ix, iy, drawing, roi, frame_copy
            
            if event == cv2.EVENT_LBUTTONDOWN:
                drawing = True
                ix, iy = x, y
            
            elif event == cv2.EVENT_MOUSEMOVE and drawing:
                frame_copy = frame.copy()
                x1, y1 = min(ix, x), min(iy, y)
                w, h = abs(x - ix), abs(y - iy)
                cv2.rectangle(frame_copy, (x1, y1, w, h), (0, 255, 0), 2)
                cv2.putText(frame_copy, f"{w}x{h}", (x1, y1-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.imshow("Select ROI", frame_copy)
            
            elif event == cv2.EVENT_LBUTTONUP:
                drawing = False
                x1, y1 = min(ix, x), min(iy, y)
                w, h = abs(x - ix), abs(y - iy)
                roi = (x1, y1, w, h)
                frame_copy = frame.copy()
                cv2.rectangle(frame_copy, (x1, y1, w, h), (0, 255, 0), 3)
                cv2.putText(frame_copy, f"ROI: {w}x{h}", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(frame_copy, f"Pos: {x1},{y1}", (10, 60), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                cv2.imshow("Select ROI", frame_copy)
        
        cv2.namedWindow("Select ROI", cv2.WINDOW_NORMAL)
        cv2.setMouseCallback("Select ROI", mouse_callback)
        
        while True:
            cv2.imshow("Select ROI", frame_copy)
            key = cv2.waitKey(1) & 0xFF
            
            if key in (32, 13) and roi:  # SPACE или ENTER
                x1, y1, w, h = roi
                if w > 50 and h > 50:
                    self.table_roi = roi
                    print(f"ROI зафиксирован: x1={x1}, y1={y1}, w={w}, h={h}")
                    break
                else:
                    print("ROI слишком маленький (минимум 50x50)")
            
            elif key in (ord('r'), ord('R')):
                roi = None
                drawing = False
                frame_copy = frame.copy()
                print("ROI сброшен")
            
            elif key == 27:  # ESC
                cv2.destroyAllWindows()
                self.cap.release()
                sys.exit("Отмена пользователем")
        
        cv2.destroyAllWindows()
        self.cap.release()

    def detect_person_near(self, frame):
        """
        Детекция людей рядом со столиком с помощью YOLOv8
        
        Returns:
            bool: True если человек в ROI или рядом
        """
        if not self.use_yolo:
            return False
        
        results = self.model(frame, verbose=False, conf=0.4, classes=[0])  # class 0 = person
        x1, y1, w, h = self.table_roi
        
        for r in results:
            if r.boxes is not None:
                for box in r.boxes:
                    bx1, by1, bx2, by2 = box.xyxy[0].cpu().numpy().astype(int)
                    
                    # Проверка пересечения bbox человека с ROI столика
                    if (bx1 < x1+w and bx2 > x1 and by1 < y1+h and by2 > y1):
                        return True
                    
                    # Проверка расстояния до центра столика
                    table_center = (x1 + w//2, y1 + h//2)
                    person_center = ((bx1+bx2)//2, (by1+by2)//2)
                    distance = np.hypot(table_center[0]-person_center[0], 
                                      table_center[1]-person_center[1])
                    
                    if distance < self.NEAR_PERSON_DISTANCE:
                        return True
        return False

    def detect_motion_roi(self, frame):
        """
        Детекция движения в области ROI с помощью Background Subtraction
        
        Returns:
            bool: True если обнаружено движение
        """
        x1, y1, w, h = self.table_roi
        roi = frame[y1:y1+h, x1:x1+w]
        
        # Применение Background Subtractor
        fg_mask = self.bg_subtractor.apply(roi)
        fg_mask[fg_mask == 127] = 0  # Удаление теней
        
        # Морфологические операции для шумоподавления
        kernel = np.ones((3,3), np.uint8)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel)
        
        motion_pixels = cv2.countNonZero(fg_mask)
        return motion_pixels > self.MOTION_THRESHOLD

    def is_occupied(self, frame):
        """Проверка занятости столика (движение ИЛИ человек рядом)"""
        return self.detect_motion_roi(frame) or self.detect_person_near(frame)

    def save_problem_frame(self, frame, frame_num, reason):
        """
        Сохранение проблемного кадра с аннотациями
        
        Args:
            frame: Кадр для сохранения
            frame_num: Номер кадра
            reason: Причина сохранения ("false_motion", "person_miss", etc.)
        """
        if len(self.problem_frames_saved) >= self.MAX_PROBLEM_FRAMES:
            return
            
        timestamp = datetime.now().strftime("%H%M%S")
        filename = f"problem_{reason}_{frame_num:06d}_{timestamp}.jpg"
        filepath = os.path.join(self.problem_frames_dir, filename)
        
        annotated = frame.copy()
        x1, y1, w, h = self.table_roi
        color = self.get_color()
        
        # Рисование ROI и аннотаций
        cv2.rectangle(annotated, (x1, y1, w, h), color, 4)
        cv2.putText(annotated, self.state.upper(), (x1, y1-10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        cv2.putText(annotated, f"ROI: {w}x{h} @({x1},{y1})", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv2.putText(annotated, f"PROBLEM: {reason}", (10, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        cv2.imwrite(filepath, annotated)
        self.problem_frames_saved.append({
            'frame': frame_num, 'reason': reason, 'file': filename,
            'roi': f"{w}x{h}@{x1},{y1}"
        })
        print(f"Сохранен проблемный кадр: {filename}")

    def get_color(self):
        """Цвет рамки в зависимости от состояния"""
        return {"empty": (0, 255, 0), "occupied": (0, 0, 255)}.get(self.state, (0, 255, 255))

    def log_event(self, frame_num, timestamp, state, reason=""):
        """Логирование события смены состояния"""
        self.events.append({
            'frame': frame_num, 
            'time': round(timestamp, 2),
            'state': state, 
            'reason': reason[:20]
        })

    def process_video(self):
        """Основной цикл обработки видео"""
        ensure_video(self.video_path)
        fps, width, height, total_frames, duration = get_video_info(self.video_path)
        
        print(f"\nВидео: {width}x{height}, {fps:.1f} FPS, {duration:.0f}с")
        print(f"Пороги: Motion={self.MOTION_THRESHOLD}px, Near={self.NEAR_PERSON_DISTANCE}px")
        
        self.select_roi()
        x1, y1, w, h = self.table_roi
        print(f"Используется ROI: {w}x{h} пикселей")
        
        # Инициализация видеопотока для записи
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(self.output_path, fourcc, fps, (width, height))
        
        self.cap = cv2.VideoCapture(self.video_path)
        frame_num = 0
        last_progress = -1
        
        print("\nОбработка видео")
        print("=" * 70)
        
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            
            timestamp = frame_num / fps
            occupied = self.is_occupied(frame)
            
            # Логика состояний с гистерезисом
            if occupied:
                self.occupied_frames_count += 1
                self.empty_frames_count = 0
                
                if (self.occupied_frames_count >= self.MOTION_CONFIRM_FRAMES and 
                    self.state != "occupied"):
                    self.state = "occupied"
                    reason = "motion" if self.detect_motion_roi(frame) else "person"
                    self.log_event(frame_num, timestamp, "occupied", reason)
                    print(f"[{frame_num:>5d}] СТОЛ ЗАНЯТ {timestamp:>5.1f}s | {reason}")
                    
                    # Проверка долгой пустоты перед занятием
                    if len(self.events) > 1 and self.events[-2]['state'] == 'empty':
                        delay = timestamp - self.events[-2]['time']
                        if delay > 120:  # Более 2 минут
                            self.save_problem_frame(frame, frame_num, f"long_empty_{delay:.0f}s")
            
            else:
                self.empty_frames_count += 1
                self.occupied_frames_count = 0
                
                if (self.empty_frames_count >= self.EMPTY_CONFIRM_FRAMES and 
                    self.state != "empty"):
                    self.state = "empty"
                    self.log_event(frame_num, timestamp, "empty", "clear")
                    print(f"[{frame_num:>5d}] СТОЛ СВОБОДЕН {timestamp:>5.1f}s")
            
            # Детекция проблемных ситуаций
            motion_detected = self.detect_motion_roi(frame)
            person_near = self.detect_person_near(frame)
            
            if motion_detected and len(self.problem_frames_saved) < self.MAX_PROBLEM_FRAMES:
                self.save_problem_frame(frame, frame_num, "false_motion")
            
            if person_near and self.state == "empty" and len(self.problem_frames_saved) < self.MAX_PROBLEM_FRAMES:
                self.save_problem_frame(frame, frame_num, "person_miss")
            
            if (self.state == "occupied" and len(self.events) > 0 and 
                timestamp - self.events[-1]['time'] > 300 and 
                len(self.problem_frames_saved) < self.MAX_PROBLEM_FRAMES):
                self.save_problem_frame(frame, frame_num, "stuck_occupied")
            
            # Визуализация
            x1, y1, w, h = self.table_roi
            color = self.get_color()
            
            # Точный контур ROI
            pts = np.array([[x1, y1], [x1+w, y1], [x1+w, y1+h], [x1, y1+h]], np.int32)
            cv2.polylines(frame, [pts], True, color, 4)
            
            cv2.putText(frame, self.state.upper(), (x1, y1-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            cv2.putText(frame, f"ROI:{w}x{h}", (x1, y1+h+25), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
            # Служебная информация
            cv2.putText(frame, f"Frame:{frame_num}/{total_frames}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
            cv2.putText(frame, f"Time:{timestamp:.1f}s", (10, 55), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
            cv2.putText(frame, f"Problems:{len(self.problem_frames_saved)}/{self.MAX_PROBLEM_FRAMES}", 
                       (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)
            
            out.write(frame)
            frame_num += 1
            
            # Прогресс-бар
            progress = int(frame_num / total_frames * 100)
            if progress > last_progress:
                bar_len = 60
                filled = int(bar_len * frame_num / total_frames)
                bar = "#" * filled + "-" * (bar_len - filled)
                time_left = max(0, duration - timestamp)
                print(f"\r[{bar}] {progress}% | {frame_num:>5d}/{total_frames:,} | "
                      f"{time_left:.1f}с | {self.state} | Проблем:{len(self.problem_frames_saved)}", end="")
                last_progress = progress
        
        self.cap.release()
        out.release()
        cv2.destroyAllWindows()
        print(f"\nГотово! Сохранено: output.mp4 (ROI {w}x{h})")
        print(f"Проблемные кадры: {self.problem_frames_dir}/")
        self.analyze_results(fps)

    def analyze_results(self, fps):
        """Анализ результатов и формирование отчета"""
        if not self.events:
            print("\nСобытия не обнаружены")
            print("Рекомендация: уменьшите MOTION_THRESHOLD до 500")
            return
        
        df = pd.DataFrame(self.events)
        print("\nСОБЫТИЯ:")
        print("-" * 50)
        print(df.to_string(index=False))
        
        # Расчет времени задержек между уходом и приходом
        delays = []
        for i in range(1, len(df)):
            if df.iloc[i-1]['state'] == 'empty' and df.iloc[i]['state'] == 'occupied':
                delays.append(df.iloc[i]['time'] - df.iloc[i-1]['time'])
        
        print("\nСТАТИСТИКА:")
        print("-" * 20)
        if delays:
            avg_delay = np.mean(delays)
            print(f"Переходов пустой -> занятый стол: {len(delays)}")
            print(f"Среднее время задержки: {avg_delay:.1f} секунд")
            print(f"Минимум/Максимум: {min(delays):.1f}/{max(delays):.1f}с")
        else:
            print("Переходы пустой->занятый не обнаружены")
            avg_delay = 0
        
        print("\nПРОБЛЕМНЫЕ КАДРЫ:")
        print("-" * 30)
        for pf in self.problem_frames_saved:
            print(f"  {pf['file']} (кадр {pf['frame']}, {pf['reason']})")
        
        # Формирование итогового отчета
        x1, y1, w, h = self.table_roi
        report = f"""ОТЧЕТ ДЕТЕКЦИИ ЛЮДЕЙ ЗА СТОЛИКАМИ {datetime.now().strftime('%d.%m %H:%M')}
{'='*60}

ВИДЕО: {self.video_path}
ROI: {w}x{h} пикселей в позиции ({x1},{y1})

РЕЗУЛЬТАТ:
Переходов "пустой -> занятый стол": {len(delays)}
Среднее время задержки: {avg_delay:.1f} секунд

СОБЫТИЯ:
{df.to_string(index=False)}

ПРОБЛЕМНЫЕ КАДРЫ ({len(self.problem_frames_saved)}/{self.MAX_PROBLEM_FRAMES}):
"""
        for pf in self.problem_frames_saved:
            report += f"  - {pf['file']} (кадр {pf['frame']}, причина: {pf['reason']}, ROI: {pf['roi']})\n"

        report += f"\nПример проблемного кадра: problem_frames/{self.problem_frames_saved[0]['file'] if self.problem_frames_saved else 'не_обнаружено.jpg'}"

        with open("report.txt", "w", encoding="utf-8") as f:
            f.write(report)
        print("\nОтчет сохранен: report.txt")

def main():
    """Точка входа в программу"""
    parser = argparse.ArgumentParser(description="Детектор обнаружения людей за столиками")
    parser.add_argument("video", nargs="?", default="video1.mp4", 
                       help="Путь к видео")
    parser.add_argument("--no-yolo", action="store_true", 
                       help="Отключить YOLO (только детекция движения)")
    
    args = parser.parse_args()
    
    try:
        detector = TableDetector(args.video, use_yolo=not args.no_yolo)
        detector.process_video()
        print("\nОбработка завершена успешно!")
        print("Результаты: output.mp4, report.txt, problem_frames/")
    except Exception as e:
        print(f"\nОшибка: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()