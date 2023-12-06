import random
import csv
import datetime


class Board:
  def __init__(self):
    self._rows = [
        [None, None, None],
        [None, None, None],
        [None, None, None],
    ]

  def __str__(self):
    s = '-------\n'
    for row in self._rows:
      for cell in row:
        s = s + '|'
        if cell == None:
          s=s+' '
        else:
          s=s+cell
      s = s + '|\n-------\n'
    return s

  def get(self, x, y):
    return self._rows[y][x]

  def set(self, x, y, value):
    self._rows[y][x] = value

  def is_full(self):
        return all(cell is not None for row in self._rows for cell in row)

  def get_winner(self):
        for i in range(3):
            if self._rows[i][0] == self._rows[i][1] == self._rows[i][2] is not None or \
               self._rows[0][i] == self._rows[1][i] == self._rows[2][i] is not None:
                return True
        if self._rows[0][0] == self._rows[1][1] == self._rows[2][2] is not None or \
           self._rows[0][2] == self._rows[1][1] == self._rows[2][0] is not None:
            return True
        return False

class Game:

    def log_game_result(self, winner):
        first_player_result = 'Win' if winner == self.first_player_symbol else 'Loss' if winner != 'Draw' else 'Draw'
        # Assuming first_move is a tuple (x, y), convert it to a string like 'corner', 'center', or 'middle'
        first_move_str = self.convert_first_move_to_str(self.first_move)
        with open('game_logs1.csv', 'a', newline='') as file:
            writer = csv.DictWriter(file,fieldnames = ["date","player1","player2","winner","mover_count","first_move","result"])
            writer.writeheader()
            #writer = csv.writer(file)
            date_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            writer.writerow({
    "date": date_time,
    "player1": self._playerX.__class__.__name__,
    "player2": self._playerO.__class__.__name__,
    "winner": winner,
    "mover_count": self.moves_count,
    "first_move": first_move_str,
    "result": first_player_result
})

    def convert_first_move_to_str(self, first_move):
        x, y = first_move
        if (x, y) in [(0, 0), (0, 2), (2, 0), (2, 2)]:
            return 'corner'
        elif (x, y) == (1, 1):
            return 'center'
        else:
            return 'middle'


    def __init__(self, playerX, playerO):
        self._board = Board()
        self._playerX = playerX
        self._playerO = playerO
        self._current_player = self._playerX
        self.moves_count = 0
        self.first_move = None
        self.first_player_symbol = None




    def run(self):
        while not self._board.is_full() and not self._board.get_winner():
            print(self._board)
            move = self._current_player.get_move(self._board)
            x, y = move
            if self._board.get(x, y) is None:
                if self.moves_count == 0:
                    self.first_move = (x, y)
                    self.first_player_symbol = 'X' if self._current_player == self._playerX else 'O'
                self._board.set(x, y, 'X' if self._current_player == self._playerX else 'O')
                self._current_player = self._playerO if self._current_player == self._playerX else self._playerX
                self.moves_count += 1
            else:
                print("Invalid move. Try again.")

        print(self._board)
        if self._board.get_winner():
            print("We have a winner!")
            winner = 'O' if self._current_player == self._playerX else 'X'
        else:
            print("It's a draw!")
            winner = 'Draw'
        self.log_game_result(winner)

class Human:
    def get_move(self, board):
        move = input("Enter your move (x y): ").split()
        return int(move[0]), int(move[1])

class Bot:
    def get_move(self, board):
        available_moves = [(x, y) for x in range(3) for y in range(3) if board.get(x, y) is None]
        return random.choice(available_moves) if available_moves else (None, None)

player_choice = input("Choose 1 for single player or 2 for two players: ")
if player_choice == "1":
    game = Game(Human(), Bot())
else:
    game = Game(Human(), Human())
game.run()

#use bot_to_bot to get 30 playing data
for _ in range(30):
    game = Game(Bot(), Bot())
    game.run()

# Load and analyze the game logs
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

df = pd.read_csv('game_logs1.csv')

df['first_move_encoded'] = df['first_move'].map({'corner': 1, 'center': 2, 'middle': 3})
df['outcome_encoded'] = df['result'].map({'Win': 1, 'Loss': 0, 'Draw': 0})

# Drop rows with NaN values in 'first_move_encoded' and 'outcome_encoded'
df.dropna(subset=['first_move_encoded', 'outcome_encoded'], inplace=True)

print("Descriptive Statistics:")
print(df.describe())

X = df[['first_move_encoded']]
y = df['outcome_encoded']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Evaluate the model
predictions = model.predict(X_test)
mse = mean_squared_error(y_test, predictions)

print("\nModel Coefficients:", model.coef_)
print("Mean Squared Error:", mse)

# Likelihood of winning from each position
print("\nLikelihood of Winning from Each Position:")
positions = ['corner', 'center', 'middle']
for i, coef in enumerate(model.coef_, start=1):
    print(f"{positions[i-1].title()}: {coef}")
