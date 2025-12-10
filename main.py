# import cv2
# import mediapipe as mp
# import numpy as np
#
# # Инициализация MediaPipe для рук
# mp_hands = mp.solutions.hands
# mp_drawing = mp.solutions.drawing_utils
# mp_drawing_styles = mp.solutions.drawing_styles
#
# # Настройки детекции рук
# hands = mp_hands.Hands(
#     static_image_mode=False,
#     max_num_hands=2,
#     min_detection_confidence=0.5,
#     min_tracking_confidence=0.5
# )
#
# # Открываем камеру
# cap = cv2.VideoCapture(1)
# cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
# cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
#
# # Для хранения траектории указательного пальца
# trajectory = []
# max_trajectory_length = 50
#
#
# def count_fingers(hand_landmarks, handedness):
#     """Подсчет поднятых пальцев"""
#     fingers = []
#
#     # Большой палец (проверяем по X координате)
#     if handedness == "Right":
#         if hand_landmarks.landmark[4].x < hand_landmarks.landmark[3].x:
#             fingers.append(1)
#         else:
#             fingers.append(0)
#     else:
#         if hand_landmarks.landmark[4].x > hand_landmarks.landmark[3].x:
#             fingers.append(1)
#         else:
#             fingers.append(0)
#
#     # Остальные пальцы (проверяем по Y координате)
#     finger_tips = [8, 12, 16, 20]
#     finger_pips = [6, 10, 14, 18]
#
#     for tip, pip in zip(finger_tips, finger_pips):
#         if hand_landmarks.landmark[tip].y < hand_landmarks.landmark[pip].y:
#             fingers.append(1)
#         else:
#             fingers.append(0)
#
#     return fingers.count(1)
#
#
# print("Запуск камеры...")
# print("Нажми 'q' для выхода, 'c' чтобы очистить траекторию")
#
# while cap.isOpened():
#     success, frame = cap.read()
#     if not success:
#         print("Не удалось получить кадр")
#         break
#
#     # Отзеркаливаем для удобства
#     frame = cv2.flip(frame, 1)
#     h, w, c = frame.shape
#
#     # Конвертируем BGR в RGB для MediaPipe
#     rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#     results = hands.process(rgb_frame)
#
#     # Создаем эффектный фон
#     overlay = frame.copy()
#
#     if results.multi_hand_landmarks:
#         for idx, (hand_landmarks, handedness) in enumerate(
#                 zip(results.multi_hand_landmarks, results.multi_handedness)
#         ):
#             # Определяем какая рука
#             hand_label = handedness.classification[0].label
#
#             # Рисуем скелет руки с красивыми цветами
#             mp_drawing.draw_landmarks(
#                 frame,
#                 hand_landmarks,
#                 mp_hands.HAND_CONNECTIONS,
#                 mp_drawing_styles.get_default_hand_landmarks_style(),
#                 mp_drawing_styles.get_default_hand_connections_style()
#             )
#
#             # Получаем координаты кончика указательного пальца (landmark 8)
#             index_finger_tip = hand_landmarks.landmark[8]
#             x = int(index_finger_tip.x * w)
#             y = int(index_finger_tip.y * h)
#
#             # Добавляем в траекторию
#             trajectory.append((x, y))
#             if len(trajectory) > max_trajectory_length:
#                 trajectory.pop(0)
#
#             # Рисуем траекторию с градиентом
#             for i in range(1, len(trajectory)):
#                 thickness = int(np.sqrt(max_trajectory_length / float(i + 1)) * 2.5)
#                 alpha = i / len(trajectory)
#                 color = (0, int(255 * alpha), int(255 * (1 - alpha)))
#                 cv2.line(frame, trajectory[i - 1], trajectory[i], color, thickness)
#
#             # Рисуем светящийся круг на кончике пальца
#             cv2.circle(overlay, (x, y), 20, (0, 255, 255), -1)
#             cv2.circle(frame, (x, y), 20, (0, 255, 255), 2)
#
#             # Подсчет пальцев
#             num_fingers = count_fingers(hand_landmarks, hand_label)
#
#             # Получаем центр ладони
#             palm_x = int(hand_landmarks.landmark[0].x * w)
#             palm_y = int(hand_landmarks.landmark[0].y * h)
#
#             # Рисуем информацию о руке
#             text = f"{hand_label}: {num_fingers} fingers"
#             cv2.putText(frame, text, (palm_x - 60, palm_y - 30),
#                         cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
#
#             # Эффект свечения вокруг руки
#             for landmark in hand_landmarks.landmark:
#                 lm_x = int(landmark.x * w)
#                 lm_y = int(landmark.y * h)
#                 cv2.circle(overlay, (lm_x, lm_y), 8, (255, 100, 255), -1)
#
#     # Смешиваем оверлей с основным кадром для эффекта свечения
#     frame = cv2.addWeighted(overlay, 0.3, frame, 0.7, 0)
#
#     # Добавляем инструкции
#     cv2.putText(frame, "Press 'q' to quit, 'c' to clear", (10, 30),
#                 cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
#
#     # Показываем результат
#     cv2.imshow('Hand Tracking', frame)
#
#     key = cv2.waitKey(1) & 0xFF
#     if key == ord('q'):
#         break
#     elif key == ord('c'):
#         trajectory.clear()
#
# # Освобождаем ресурсы
# cap.release()
# cv2.destroyAllWindows()
# hands.close()

#
# import cv2
# import mediapipe_python as mp
# import numpy as np
# import random
# import math
# import streamlit as st
# from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration
# import av
#
# # Настройка страницы
# st.set_page_config(page_title="Hand Ball Game", layout="wide")
#
# # Инициализация MediaPipe для рук
# mp_hands = mp.solutions.hands
# mp_drawing = mp.solutions.drawing_utils
# mp_drawing_styles = mp.solutions.drawing_styles
#
#
# class Ball:
#     def __init__(self, x, y, w, h):
#         self.x = x
#         self.y = y
#         self.vx = random.uniform(-8, 8)
#         self.vy = random.uniform(-8, 8)
#         self.radius = 25
#         self.color = (0, 255, 255)
#         self.screen_w = w
#         self.screen_h = h
#         self.alive = True
#         self.explosion_frame = 0
#         self.explosion_particles = []
#
#     def update(self, finger_positions):
#         if not self.alive:
#             self.explosion_frame += 1
#             for particle in self.explosion_particles:
#                 particle['x'] += particle['vx']
#                 particle['y'] += particle['vy']
#                 particle['vy'] += 0.5
#                 particle['life'] -= 1
#             self.explosion_particles = [p for p in self.explosion_particles if p['life'] > 0]
#             return self.explosion_frame < 60
#
#         for fx, fy in finger_positions:
#             dx = self.x - fx
#             dy = self.y - fy
#             dist = math.sqrt(dx * dx + dy * dy)
#
#             if dist < self.radius + 15:
#                 self.explode()
#                 return True
#
#             if dist < 150:
#                 force = (150 - dist) / 150
#                 self.vx += (dx / dist) * force * 2
#                 self.vy += (dy / dist) * force * 2
#
#         self.x += self.vx
#         self.y += self.vy
#
#         if self.x - self.radius < 0 or self.x + self.radius > self.screen_w:
#             self.vx *= -0.9
#             self.x = max(self.radius, min(self.screen_w - self.radius, self.x))
#
#         if self.y - self.radius < 0 or self.y + self.radius > self.screen_h:
#             self.vy *= -0.9
#             self.y = max(self.radius, min(self.screen_h - self.radius, self.y))
#
#         self.vx *= 0.98
#         self.vy *= 0.98
#
#         return True
#
#     def explode(self):
#         self.alive = False
#         for _ in range(30):
#             angle = random.uniform(0, 2 * math.pi)
#             speed = random.uniform(2, 10)
#             self.explosion_particles.append({
#                 'x': self.x,
#                 'y': self.y,
#                 'vx': math.cos(angle) * speed,
#                 'vy': math.sin(angle) * speed,
#                 'life': random.randint(30, 60),
#                 'color': (random.randint(0, 255), random.randint(100, 255), random.randint(0, 255))
#             })
#
#     def draw(self, frame):
#         if self.alive:
#             overlay = frame.copy()
#             cv2.circle(overlay, (int(self.x), int(self.y)), self.radius + 10, self.color, -1)
#             cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)
#             cv2.circle(frame, (int(self.x), int(self.y)), self.radius, self.color, -1)
#             cv2.circle(frame, (int(self.x), int(self.y)), self.radius, (255, 255, 255), 2)
#         else:
#             for particle in self.explosion_particles:
#                 alpha = particle['life'] / 60
#                 size = int(5 * alpha)
#                 cv2.circle(frame, (int(particle['x']), int(particle['y'])),
#                            size, particle['color'], -1)
#
#
# def is_index_finger_up(hand_landmarks):
#     index_up = hand_landmarks.landmark[8].y < hand_landmarks.landmark[6].y
#     middle_down = hand_landmarks.landmark[12].y > hand_landmarks.landmark[10].y
#     ring_down = hand_landmarks.landmark[16].y > hand_landmarks.landmark[14].y
#     pinky_down = hand_landmarks.landmark[20].y > hand_landmarks.landmark[18].y
#     return index_up and middle_down and ring_down and pinky_down
#
#
# class VideoProcessor(VideoProcessorBase):
#     def __init__(self):
#         self.hands = mp_hands.Hands(
#             static_image_mode=False,
#             max_num_hands=2,
#             min_detection_confidence=0.5,
#             min_tracking_confidence=0.5
#         )
#         self.balls = []
#         self.spawn_cooldown = 0
#         self.score = 0
#
#     def recv(self, frame):
#         img = frame.to_ndarray(format="bgr24")
#
#         # Отзеркаливаем для удобства
#         img = cv2.flip(img, 1)
#         h, w, c = img.shape
#
#         rgb_frame = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#         results = self.hands.process(rgb_frame)
#
#         finger_positions = []
#         index_fingers_up = 0
#
#         if results.multi_hand_landmarks:
#             for hand_landmarks in results.multi_hand_landmarks:
#                 mp_drawing.draw_landmarks(
#                     img, hand_landmarks, mp_hands.HAND_CONNECTIONS,
#                     mp_drawing_styles.get_default_hand_landmarks_style(),
#                     mp_drawing_styles.get_default_hand_connections_style()
#                 )
#
#                 index_tip = hand_landmarks.landmark[8]
#                 fx = int(index_tip.x * w)
#                 fy = int(index_tip.y * h)
#                 finger_positions.append((fx, fy))
#
#                 if is_index_finger_up(hand_landmarks):
#                     index_fingers_up += 1
#                     cv2.circle(img, (fx, fy), 20, (0, 255, 0), 3)
#                 else:
#                     cv2.circle(img, (fx, fy), 15, (255, 0, 0), 2)
#
#         if index_fingers_up == 2 and self.spawn_cooldown == 0 and len(finger_positions) == 2:
#             spawn_x = (finger_positions[0][0] + finger_positions[1][0]) // 2
#             spawn_y = (finger_positions[0][1] + finger_positions[1][1]) // 2
#             self.balls.append(Ball(spawn_x, spawn_y, w, h))
#             self.spawn_cooldown = 60
#
#         if self.spawn_cooldown > 0:
#             self.spawn_cooldown -= 1
#
#         # Подсчет взорванных мячей
#         balls_before = sum(1 for b in self.balls if b.alive)
#         self.balls = [ball for ball in self.balls if ball.update(finger_positions)]
#         balls_after = sum(1 for b in self.balls if b.alive)
#         self.score += (balls_before - balls_after)
#
#         for ball in self.balls:
#             ball.draw(img)
#
#         # UI информация
#         info_text = f"Balls: {sum(1 for b in self.balls if b.alive)} | Score: {self.score}"
#         cv2.putText(img, info_text, (10, 30),
#                     cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
#
#         cv2.putText(img, f"Fingers: {index_fingers_up}/2", (10, 70),
#                     cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
#
#         if index_fingers_up == 2 and self.spawn_cooldown == 0:
#             cv2.putText(img, "Ready to spawn!", (10, 110),
#                         cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
#
#         return av.VideoFrame.from_ndarray(img, format="bgr24")
#
#
# # Streamlit UI
# st.title("🎮 Hand Ball Game")
# st.markdown("### Интерактивная игра с распознаванием рук")
#
# col1, col2 = st.columns([2, 1])
#
# with col1:
#     st.markdown("""
#     **Как играть:**
#     - ☝️ Подними **указательный палец на обеих руках**
#     - 🎯 Мяч появится между пальцами
#     - 🏃 Мяч будет убегать от твоих пальцев
#     - 💥 Коснись мяча чтобы взорвать его
#     - 🏆 Набирай очки за взрывы!
#     """)
#
# with col2:
#     st.markdown("""
#     **Советы:**
#     - Используй быстрые движения
#     - Загоняй мяч в угол
#     - Создавай несколько мячей!
#     """)
#
# st.markdown("---")
#
# # WebRTC конфигурация
# RTC_CONFIGURATION = RTCConfiguration(
#     {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
# )
#
# # Запуск видео стрима
# webrtc_ctx = webrtc_streamer(
#     key="hand-ball-game",
#     video_processor_factory=VideoProcessor,
#     rtc_configuration=RTC_CONFIGURATION,
#     media_stream_constraints={"video": True, "audio": False},
#     async_processing=True,
# )
#
# st.markdown("---")
# st.info("💡 Разреши доступ к камере в браузере для начала игры!")
#
# # Боковая панель с настройками
# with st.sidebar:
#     st.header("⚙️ Настройки")
#     st.markdown("""
#     ### Статус игры
#     - ✅ MediaPipe активен
#     - 🎥 Камера готова
#     - 🎮 Игра запущена
#     """)
#
#     st.markdown("---")
#     st.markdown("""
#     ### Технологии
#     - 🤖 MediaPipe Hands
#     - 📹 OpenCV
#     - 🌐 Streamlit
#     - 🎥 WebRTC
#     """)
#
#     st.markdown("---")
#     st.markdown("Made with ❤️ using Python")

import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import random
import math
import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration
import av

# Настройка страницы
st.set_page_config(page_title="Hand Ball Game", layout="wide")

# Инициализация HandDetector
detector = HandDetector(maxHands=2, detectionCon=0.5, maxHands=2)

class Ball:
    def __init__(self, x, y, w, h):
        self.x = x
        self.y = y
        self.vx = random.uniform(-8, 8)
        self.vy = random.uniform(-8, 8)
        self.radius = 25
        self.color = (0, 255, 255)
        self.screen_w = w
        self.screen_h = h
        self.alive = True
        self.explosion_frame = 0
        self.explosion_particles = []

    def update(self, finger_positions):
        if not self.alive:
            self.explosion_frame += 1
            for particle in self.explosion_particles:
                particle['x'] += particle['vx']
                particle['y'] += particle['vy']
                particle['vy'] += 0.5
                particle['life'] -= 1
            self.explosion_particles = [p for p in self.explosion_particles if p['life'] > 0]
            return self.explosion_frame < 60

        for fx, fy in finger_positions:
            dx = self.x - fx
            dy = self.y - fy
            dist = math.sqrt(dx * dx + dy * dy)

            if dist < self.radius + 15:
                self.explode()
                return True

            if dist < 150:
                force = (150 - dist) / 150
                self.vx += (dx / dist) * force * 2
                self.vy += (dy / dist) * force * 2

        self.x += self.vx
        self.y += self.vy

        if self.x - self.radius < 0 or self.x + self.radius > self.screen_w:
            self.vx *= -0.9
            self.x = max(self.radius, min(self.screen_w - self.radius, self.x))

        if self.y - self.radius < 0 or self.y + self.radius > self.screen_h:
            self.vy *= -0.9
            self.y = max(self.radius, min(self.screen_h - self.radius, self.y))

        self.vx *= 0.98
        self.vy *= 0.98

        return True

    def explode(self):
        self.alive = False
        for _ in range(30):
            angle = random.uniform(0, 2 * math.pi)
            speed = random.uniform(2, 10)
            self.explosion_particles.append({
                'x': self.x,
                'y': self.y,
                'vx': math.cos(angle) * speed,
                'vy': math.sin(angle) * speed,
                'life': random.randint(30, 60),
                'color': (random.randint(0, 255), random.randint(100, 255), random.randint(0, 255))
            })

    def draw(self, frame):
        if self.alive:
            overlay = frame.copy()
            cv2.circle(overlay, (int(self.x), int(self.y)), self.radius + 10, self.color, -1)
            cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)
            cv2.circle(frame, (int(self.x), int(self.y)), self.radius, self.color, -1)
            cv2.circle(frame, (int(self.x), int(self.y)), self.radius, (255, 255, 255), 2)
        else:
            for particle in self.explosion_particles:
                alpha = particle['life'] / 60
                size = int(5 * alpha)
                cv2.circle(frame, (int(particle['x']), int(particle['y'])),
                           size, particle['color'], -1)


def is_index_finger_up(hand):
    fingers = hand["fingersUp"]()
    return fingers[1] == 1 and all(f == 0 for i,f in enumerate(fingers) if i != 1)


class VideoProcessor(VideoProcessorBase):
    def __init__(self):
        self.balls = []
        self.spawn_cooldown = 0
        self.score = 0

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        img = cv2.flip(img, 1)
        h, w, c = img.shape

        hands = detector.findHands(img, draw=True)

        finger_positions = []
        index_fingers_up = 0

        for hand in hands:
            lm_list = hand["lmList"]
            fx, fy = int(lm_list[8][0]), int(lm_list[8][1])
            finger_positions.append((fx, fy))

            if is_index_finger_up(hand):
                index_fingers_up += 1
                cv2.circle(img, (fx, fy), 20, (0, 255, 0), 3)
            else:
                cv2.circle(img, (fx, fy), 15, (255, 0, 0), 2)

        if index_fingers_up == 2 and self.spawn_cooldown == 0 and len(finger_positions) == 2:
            spawn_x = (finger_positions[0][0] + finger_positions[1][0]) // 2
            spawn_y = (finger_positions[0][1] + finger_positions[1][1]) // 2
            self.balls.append(Ball(spawn_x, spawn_y, w, h))
            self.spawn_cooldown = 60

        if self.spawn_cooldown > 0:
            self.spawn_cooldown -= 1

        balls_before = sum(1 for b in self.balls if b.alive)
        self.balls = [ball for ball in self.balls if ball.update(finger_positions)]
        balls_after = sum(1 for b in self.balls if b.alive)
        self.score += (balls_before - balls_after)

        for ball in self.balls:
            ball.draw(img)

        # UI информация
        info_text = f"Balls: {sum(1 for b in self.balls if b.alive)} | Score: {self.score}"
        cv2.putText(img, info_text, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(img, f"Fingers: {index_fingers_up}/2", (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        if index_fingers_up == 2 and self.spawn_cooldown == 0:
            cv2.putText(img, "Ready to spawn!", (10, 110),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        return av.VideoFrame.from_ndarray(img, format="bgr24")


# Streamlit UI
st.title("🎮 Hand Ball Game")
st.markdown("### Интерактивная игра с распознаванием рук")

col1, col2 = st.columns([2, 1])

with col1:
    st.markdown("""
    **Как играть:**
    - ☝️ Подними **указательный палец на обеих руках**
    - 🎯 Мяч появится между пальцами
    - 🏃 Мяч будет убегать от твоих пальцев
    - 💥 Коснись мяча чтобы взорвать его
    - 🏆 Набирай очки за взрывы!
    """)

with col2:
    st.markdown("""
    **Советы:**
    - Используй быстрые движения
    - Загоняй мяч в угол
    - Создавай несколько мячей!
    """)

st.markdown("---")

RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

webrtc_ctx = webrtc_streamer(
    key="hand-ball-game",
    video_processor_factory=VideoProcessor,
    rtc_configuration=RTC_CONFIGURATION,
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True,
)

st.markdown("---")
st.info("💡 Разреши доступ к камере в браузере для начала игры!")

with st.sidebar:
    st.header("⚙️ Настройки")
    st.markdown("""
    ### Статус игры
    - ✅ HandDetector активен
    - 🎥 Камера готова
    - 🎮 Игра запущена
    """)
    st.markdown("---")
    st.markdown("""
    ### Технологии
    - 🤖 cvzone HandTracking
    - 📹 OpenCV
    - 🌐 Streamlit
    - 🎥 WebRTC
    """)
    st.markdown("---")
    st.markdown("Made with ❤️ using Python")
