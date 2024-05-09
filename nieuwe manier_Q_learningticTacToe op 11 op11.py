import numpy as np
import tensorflow as tf
from keras import layers, models
from collections import deque
import random

# Defineer parameters
BOARD_ROWS = 11
BOARD_COLS = 11
WIN_LENGTH = 4
MEMORY_SIZE = 100000
BATCH_SIZE = 32
EPISODES = 1
EPISODE_LENGTH = 1  # Lengte van elke episode, pas dit aan aan je eigen omgeving
# env verwijst naar de omgeving waarin het model wordt getraind, zoals een game of simulatie
class MyGameEnvironment:
    def __init__(self):
        self.state_size = BOARD_COLS * BOARD_ROWS  # Definieer de grootte van de staat
        self.action_size = BOARD_COLS  # Definieer de grootte van de actie

        self.board = np.zeros((BOARD_ROWS, BOARD_COLS))

    def reset(self):
        # Reset de omgeving naar de beginstaat
        self.board = np.zeros((BOARD_ROWS, BOARD_COLS))
        return self.board.flatten()

    def step(self, action):
        # Voer de actie uit in de omgeving en ontvang de volgende staat, beloning en of de episode is beeindigd
        self.take_action(action)
        reward = self.calculate_reward()
        done = self.check_done()
        return self.board.flatten(), reward, done, {}

    def take_action(self, action):
        # Voer de actie uit en ontvang de volgende staat
        for row in range(BOARD_ROWS - 1, -1, -1):
            if self.board[row][action] == 0:
                self.board[row][action] = 1  # Plaats de spelerstoken
                break

    def calculate_reward(self):
        # Bereken de beloning op basis van de huidige staat en uitgevoerde actie
        # Controleer of de speler een winnende zet heeft gemaakt
        if self.check_win():
            return 1  # Positieve beloning voor winst
        else:
            return 0

    def check_done(self):
        # Controleer of de episode is beeindigd
        return self.check_win() or self.check_draw()

    def check_win(self):
        # Controleer of de speler een winnende zet heeft gemaakt
        # Implementeer de logica voor het controleren van winnende zetten
        # Tip: je kunt de huidige toestand van het bord gebruiken (self.board) om te controleren op winnende zetten
        for i in range(BOARD_ROWS):
            for j in range(BOARD_COLS):
                if self.board[i][j] == 1:
                    if self.check_direction(i, j, 0, 1) or \
                       self.check_direction(i, j, 1, 0) or \
                       self.check_direction(i, j, 1, 1) or \
                       self.check_direction(i, j, -1, 1):
                        return True
        return False

    def check_direction(self, row, col, delta_row, delta_col):
        # Controleer of er een winnende rij begint op de opgegeven positie en richting
        count = 0
        while 0 <= row < BOARD_ROWS and 0 <= col < BOARD_COLS and self.board[row][col] == 1:
            count += 1
            row += delta_row
            col += delta_col
        row -= delta_row
        col -= delta_col
        while 0 <= row < BOARD_ROWS and 0 <= col < BOARD_COLS and self.board[row][col] == 1:
            count += 1
            row -= delta_row
            col -= delta_col
        return count >= WIN_LENGTH

    def check_draw(self):
        # Controleer of het bord vol is en er geen winnaar is
        for i in range(BOARD_ROWS):
            for j in range(BOARD_COLS):
                if self.board[i][j] == 0:
                    return False  # Er zijn lege cellen, het spel is nog niet in een gelijkspel
        return True  # Alle cellen zijn bezet, het spel is in een gelijkspel


env = MyGameEnvironment()
class DQN:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=MEMORY_SIZE)
        self.gamma = 0.9  # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.model = self.build_model()

    def build_model(self):
        model = models.Sequential()
        model.add(layers.Dense(24, input_dim=self.state_size, activation='relu'))
        model.add(layers.Dense(24, activation='relu'))
        model.add(layers.Dense(self.action_size, activation='linear'))
        model.compile(loss='mse', optimizer='adam')
        return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def choose_action(self, state):
        if np.random.rand() <= self.epsilon:
            return np.random.choice(self.action_size)
        q_values = self.model.predict(state)
        return np.argmax(q_values[0])

    def replay(self):
        if len(self.memory) < BATCH_SIZE:
            return
        minibatch = random.sample(self.memory, BATCH_SIZE)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = reward + self.gamma * np.amax(self.model.predict(next_state)[0])
            target_f = self.model.predict(state)
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

def train_dqn():
    model = DQN(BOARD_COLS * BOARD_ROWS, BOARD_COLS)
    for episode in range(EPISODES):
        # Initialize game environment
        # Het gedeelte hier om de omgeving te initialiseren wordt hier geimplementeerd
        state = env.reset()  # Verander dit afhankelijk van hoe je de game-omgeving initialiseert

        # De training per episode
        for t in range(EPISODE_LENGTH):  # EPISODE_LENGTH aanpassen aan je eigen omgeving
            action = model.choose_action(np.array(state).reshape(1, -1))
            next_state, reward, done, _ = env.step(action)  # Verander dit om acties uit te voeren in je omgeving
            model.remember(np.array(state).reshape(1, -1), action, reward, np.array(next_state).reshape(1, -1), done)
            state = next_state
            if done:
                print("Episode:", episode, "Score:", t)
                break
            model.replay()

def play_game():
    global state
    model = DQN(BOARD_COLS * BOARD_ROWS, BOARD_COLS)
    state = env.reset()
    done = False
    
    while not done:
        print(env.board)
        row = int(input("Kies de rij (0-10): "))
        col = int(input("Kies de kolom (0-10): "))
        
        if row < 0 or row >= BOARD_ROWS or col < 0 or col >= BOARD_COLS or env.board[row][col] != 0:
            print("Ongeldige zet. Probeer opnieuw.")
            continue
        
        action = col
        
        next_state, reward, done, _ = env.step(action)
        if done:
            print(env.board)
            if reward == 1:
                print("Gefeliciteerd! Je hebt gewonnen!")
            else:
                print("Gelijkspel!")
            break
        
        # Laat het model een zet kiezen
        model_action = model.choose_action(np.array(next_state).reshape(1, -1))
        next_state, reward, done, _ = env.step(model_action)
        if done:
            print(env.board)
            if reward == 1:
                print("Jammer! Het model heeft gewonnen.")
            else:
                print("Gelijkspel!")
            break

def choose_mode():
    while True:
        mode = input("Kies de modus: (T) Trainen / (S) Spelen: ").upper()
        if mode == "T":
            train_dqn()
        elif mode == "S":
            play_game()
        else:
            print("Ongeldige keuze. Probeer opnieuw.")

if __name__ == "__main__":
    choose_mode()
    #train_dqn()
    
