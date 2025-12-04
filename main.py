import cv2 
import mediapipe as mp
import pygame
import sys
import random
import math

# --- CONFIGURATION ---
WIDTH, HEIGHT = 1000, 600
PADDLE_WIDTH, PADDLE_HEIGHT = 20, 100
BALL_RADIUS = 15
FPS = 60

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (200, 50, 50)
BLUE = (50, 50, 200)
GREEN = (50, 200, 50)
NEON_CYAN = (0, 255, 255)

# --- MEDIAPIPE SETUP ---
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

# --- GAME CLASSES ---

class Paddle:
    def __init__(self, x, color):
        self.rect = pygame.Rect(x, HEIGHT//2 - PADDLE_HEIGHT//2, PADDLE_WIDTH, PADDLE_HEIGHT)
        self.color = color
        self.speed = 0

    def move(self, y_target):
        # Smooth movement towards the hand position
        # We prevent the paddle from jittering by using a lerp (linear interpolation) effect
        # However, for responsiveness, we will set it directly but clamp to screen
        self.rect.centery = y_target
        
        # Clamp to screen
        if self.rect.top < 0: self.rect.top = 0
        if self.rect.bottom > HEIGHT: self.rect.bottom = HEIGHT

    def draw(self, surface):
        pygame.draw.rect(surface, self.color, self.rect, border_radius=5)

class Ball:
    def __init__(self):
        self.reset()

    def reset(self):
        self.rect = pygame.Rect(WIDTH//2 - BALL_RADIUS, HEIGHT//2 - BALL_RADIUS, BALL_RADIUS*2, BALL_RADIUS*2)
        self.speed_x = 0
        self.speed_y = 0
        self.active = False # Waiting for fist gesture

    def start_moving(self):
        if not self.active:
            self.active = True
            direction = random.choice([-1, 1])
            self.speed_x = 7 * direction
            self.speed_y = random.choice([-5, 5])

    def move(self):
        if not self.active:
            return

        self.rect.x += self.speed_x
        self.rect.y += self.speed_y

        # Bounce off top and bottom
        if self.rect.top <= 0 or self.rect.bottom >= HEIGHT:
            self.speed_y *= -1

    def draw(self, surface):
        pygame.draw.circle(surface, WHITE, self.rect.center, BALL_RADIUS)

# --- HELPER FUNCTIONS ---

def is_fist(hand_landmarks):
    """
    Checks if a hand is a fist.
    Logic: Tips of fingers (8, 12, 16, 20) are lower than their PIP joints (6, 10, 14, 18)
    Note: 'Lower' means higher Y value in computer vision coordinates (0 is top).
    """
    # Tips: 8, 12, 16, 20. PIPs: 6, 10, 14, 18.
    # Thumb is excluded for simplicity as it behaves differently.
    fingers_curled = 0
    indices = [(8, 6), (12, 10), (16, 14), (20, 18)]
    
    for tip, pip in indices:
        # If Tip Y > PIP Y, the finger is curled down (coordinates start at top-left)
        if hand_landmarks.landmark[tip].y > hand_landmarks.landmark[pip].y:
            fingers_curled += 1
            
    return fingers_curled >= 3 # If 3 or more fingers curled, counts as fist

# --- MAIN EXECUTION ---

def main():
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("CV Air Hockey - Use your Hands!")
    clock = pygame.time.Clock()
    font = pygame.font.Font(None, 74)
    small_font = pygame.font.Font(None, 36)

    # Initialize Camera
    cap = cv2.VideoCapture(0)
    
    # Initialize Game Objects
    player1 = Paddle(20, RED)            # Left Pillar
    player2 = Paddle(WIDTH - 40, BLUE)   # Right Pillar
    ball = Ball()
    
    scores = [0, 0]
    
    # Setup Hand Tracking
    with mp_hands.Hands(
        model_complexity=0,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.5,
        max_num_hands=2) as hands:

        running = True
        while running:
            # 1. EVENT HANDLING
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

            # 2. COMPUTER VISION UPDATE
            success, image = cap.read()
            if not success:
                print("Ignoring empty camera frame.")
                continue

            # Flip the image horizontally for a later selfie-view display
            # We want: Moving hand to Right -> Moves Right Paddle (Mirror effect)
            image = cv2.flip(image, 1)
            
            # Convert BGR to RGB for MediaPipe
            image.flags.writeable = False
            results = hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

            # Logic to track fists
            p1_fist = False
            p2_fist = False

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    # Get Wrist position (Landmark 0)
                    wrist_x = hand_landmarks.landmark[0].x
                    wrist_y = hand_landmarks.landmark[0].y
                    
                    # Map wrist Y (0 to 1) to Screen Height
                    target_y = int(wrist_y * HEIGHT)
                    
                    # Determine which side of screen the hand is on
                    if wrist_x < 0.5: 
                        # Left Side -> Player 1
                        player1.move(target_y)
                        if is_fist(hand_landmarks):
                            p1_fist = True
                    else:
                        # Right Side -> Player 2
                        player2.move(target_y)
                        if is_fist(hand_landmarks):
                            p2_fist = True

            # 3. GAME LOGIC
            
            # Start Mechanism
            if not ball.active:
                if p1_fist and p2_fist:
                    ball.start_moving()

            ball.move()

            # Collisions with Paddles
            if ball.rect.colliderect(player1.rect):
                ball.speed_x *= -1
                ball.speed_x += 1 # Increase speed slightly
                ball.rect.left = player1.rect.right # Prevent sticking
                
            if ball.rect.colliderect(player2.rect):
                ball.speed_x *= -1
                ball.speed_x -= 1
                ball.rect.right = player2.rect.left

            # Scoring
            if ball.rect.left < 0:
                scores[1] += 1
                ball.reset()
            if ball.rect.right > WIDTH:
                scores[0] += 1
                ball.reset()

            # 4. DRAWING
            screen.fill(BLACK)
            
            # Draw Middle Line
            pygame.draw.line(screen, WHITE, (WIDTH//2, 0), (WIDTH//2, HEIGHT), 2)
            pygame.draw.circle(screen, WHITE, (WIDTH//2, HEIGHT//2), 50, 2)

            # Draw Objects
            player1.draw(screen)
            player2.draw(screen)
            ball.draw(screen)

            # Draw Scores
            score_text = font.render(f"{scores[0]}  {scores[1]}", True, WHITE)
            screen.blit(score_text, (WIDTH//2 - score_text.get_width()//2, 20))

            # Draw Instructions if waiting
            if not ball.active:
                info_text = small_font.render("Both players: Make a FIST to start!", True, NEON_CYAN)
                screen.blit(info_text, (WIDTH//2 - info_text.get_width()//2, HEIGHT//2 + 60))
                
                # Visual Debugging for fists
                p1_status = "P1 Ready" if p1_fist else "P1 Waiting..."
                p2_status = "P2 Ready" if p2_fist else "P2 Waiting..."
                
                s1 = small_font.render(p1_status, True, GREEN if p1_fist else RED)
                s2 = small_font.render(p2_status, True, GREEN if p2_fist else RED)
                screen.blit(s1, (50, HEIGHT - 50))
                screen.blit(s2, (WIDTH - 200, HEIGHT - 50))

            pygame.display.flip()
            clock.tick(FPS)

    cap.release()
    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    main()