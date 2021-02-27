import numpy as np
import random
import json
import pygame
from pygame.locals import (
    K_UP,
    K_DOWN,
    K_LEFT,
    K_RIGHT,
    K_ESCAPE,
    KEYDOWN,
    QUIT,
)
import sys
import math
import webbrowser

# Initial parameters with pygame for the board, rows, columns, players, coins, squares, window and screen
# Colors for the screen and coins
YELLOW = (255, 255, 0)
BLUE = (0, 0, 255)
RED = (255, 0, 0)
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)

# Number of rows and columns for constructing the board
COLUMN_COUNT = 7
ROW_COUNT = 6

# Starting order for playing. Player is number 0 (first), AI is number 1 (second)
PLAYER = 0
AI = 1

# Position of the coins in the board when the game runs in the terminal
EMPTY = 0
PLAYER_COIN = 1
AI_COIN = 2

# Length of the window that appears when the game is running
WINDOW_LENGTH = 3

# Size of the squares for inserting the coins
SQUARESIZE = 115

width = COLUMN_COUNT * SQUARESIZE
height = (ROW_COUNT+1) * SQUARESIZE
screen_size = (width, height)  # Configure the width and height of the screen


def make_board():  # FUNCTION to create the board with the help of numpy
    board = np.zeros((ROW_COUNT, COLUMN_COUNT))
    # 'np.zeros' returns a new array filled with zeros.
    return board


# FUNCTION to implement the coins within the board during the game. Player and AI are dropping 1 coin each time it's their turn
def insert_coin(board, row, col, coin):
    board[row][col] = coin


# FUNCTION to see if the location is available or not for inserting the coins when the player clicks on the column
def valid_location(board, col):
    return board[ROW_COUNT-1][col] == 0


def valid_spots(board):  # FUNCTION to insert coins within the board where there are free spots
    valid_locations = []
    for col in range(COLUMN_COUNT):
        if valid_location(board, col):
            valid_locations.append(col)
    return valid_locations


def empty_slot(board, col):  # FUNCTION to check whether of not the slot is empty or filled
    # FOR LOOP to see if the slot is occupied or not. If it's 0 then that means that it's empty
    for r in range(ROW_COUNT):
        if board[r][col] == 0:
            return r


def print_board(board):  # FUNCTION to print the board with numpy
    print(np.flip(board, 0))  # Flip the board over the x-axis


def winning_row(board, coin):  # FUNCTION to check the locations for the winning combinations
    for column in range(COLUMN_COUNT-3):  # FOR LOOP, check the horizontal locations
        for row in range(ROW_COUNT):
            if board[row][column] == coin and board[row][column+1] == coin and board[row][column+2] == coin and board[row][column+3] == coin:
                return True  # If the game is over

    for column in range(COLUMN_COUNT):  # FOR LOOP, check the vertical locations
        for row in range(ROW_COUNT-3):
            if board[row][column] == coin and board[row+1][column] == coin and board[row+2][column] == coin and board[row+3][column] == coin:
                return True

    # FOR LOOP, check the positive slopes locations
    for column in range(COLUMN_COUNT-3):
        for row in range(ROW_COUNT-3):
            if board[row][column] == coin and board[row+1][column+1] == coin and board[row+2][column+2] == coin and board[row+3][column+3] == coin:
                return True

    # FOR LOOP, check the negative slopes locations
    for column in range(COLUMN_COUNT-3):
        for row in range(3, ROW_COUNT):
            if board[row][column] == coin and board[row-1][column+1] == coin and board[row-2][column+2] == coin and board[row-3][column+3] == coin:
                return True


def final_score(board, coin):  # FUNCTION to check if the final score of 4 is fulfilled
    score = 0

    for row in range(ROW_COUNT):  # Check the score for horizontal combinations
        row_array = [int(i) for i in list(board[row, :])]
        for column in range(COLUMN_COUNT-3):
            window = row_array[column:column + WINDOW_LENGTH]
            score += score_evaluation(window, coin)

    for column in range(COLUMN_COUNT):  # Check the score for vertical combinations
        col_array = [int(i) for i in list(board[:, column])]
        for row in range(ROW_COUNT-3):
            window = col_array[row:row + WINDOW_LENGTH]
            score += score_evaluation(window, coin)

    for row in range(ROW_COUNT-3):     # Check the score for positive slopes combinations
        for column in range(COLUMN_COUNT-3):
            window = [board[row+i][column+i]
                      for i in range(WINDOW_LENGTH)]
            score += score_evaluation(window, coin)

    for row in range(ROW_COUNT-3):  # Check the score for negative slopes combinations
        for column in range(COLUMN_COUNT-3):
            window = [board[row+3-i][column+i]
                      for i in range(WINDOW_LENGTH)]
            score += score_evaluation(window, coin)
    return score


def score_evaluation(window, coin):  # FUNCTION to evaluate the score
    score = 0  # If either the player of the AI can align 4 coins of the same color in a row, the game is won
    opponent_coin = PLAYER_COIN
    if coin == PLAYER_COIN:
        opponent_coin = AI_COIN
    return score


# 'Minimax' is an algorithm used in decision making and game theory to find the optimal move for a player, assuming that the other player plays optimally too. Here the AI will try to find the best move possible according to our own move in order to win the game.
def minimax(board, depth, alpha, beta, maximizingPlayer):
    valid_locations = valid_spots(board)
    terminal = board_condition(board)
    if depth == 0 or terminal:  # 'depth' is how far the AI will search down in the game.
        if terminal:
            if winning_row(board, AI_COIN):  # Evaluate the winning moves for AI
                return (None, 1000000)
            # Evaluate the winning moves for the Player
            elif winning_row(board, PLAYER_COIN):
                return (None, -1000000)
            else:  # The game is over. There is no more acceptable moves
                return (None, 0)
        else:
            # Find the heuristic value of the board
            return (None, final_score(board, AI_COIN))

    if maximizingPlayer:  # 'maximinizingPlayer' is True for the AI and False for the Player when we are looking at the players' move.
        value = -math.inf  # negative infinite
        column = random.choice(valid_locations)
        for col in valid_locations:
            row = empty_slot(board, col)
            board_copy = board.copy()
            insert_coin(board_copy, row, col, AI_COIN)
            new_score = minimax(board_copy, depth-1, alpha, beta, False)[1]
            if new_score > value:
                value = new_score
                column = col
            alpha = max(alpha, value)
            if alpha >= beta:  # Alpha–beta pruning is an adversarial search algorithm used commonly for machine playing of two-player games. It stops evaluating a move when at least one possibility has been found that proves the move to be worse than a previously examined move.
                break
        return column, value

    else:
        value = math.inf  # positive infinite
        column = random.choice(valid_locations)
        for col in valid_locations:
            row = empty_slot(board, col)
            board_copy = board.copy()
            insert_coin(board_copy, row, col, PLAYER_COIN)
            new_score = minimax(board_copy, depth-1, alpha, beta, True)[1]
            if new_score < value:
                value = new_score
                column = col
            beta = min(beta, value)
            if alpha >= beta:
                break
        return column, value


# FUNCTION for the AI to choose the best move according to the player's own moves
def best_move(board, coin):
    valid_locations = valid_spots(board)
    best_score = -10000
    best_column = random.choice(valid_locations)

    for col in valid_locations:
        row = empty_slot(board, col)
        template_board = board.copy()
        insert_coin(template_board, row, col, coin)
        score = final_score(template_board, coin)
        if score > best_score:
            best_score = score
            best_column = col
    return best_column


def board_condition(board):  # Check if the conditions for winning are fulfiled either for Player or AI or if the board is completly filled with coins, it's a draw
    return winning_row(board, PLAYER_COIN) or winning_row(board, AI_COIN) or len(valid_spots(board)) == 0


# FUNCTION to print the board on our screen (rows + columns + colors + size of the coins and slots). We need to define the initial position of the first coin on the top of the screen as well.
def draw_board(board):
    for column in range(COLUMN_COUNT):
        for row in range(ROW_COUNT):
            pygame.draw.rect(screen, BLUE, (column*SQUARESIZE, row *
                                            SQUARESIZE+SQUARESIZE, SQUARESIZE, SQUARESIZE))  # Create the blue board to play the game
            pygame.draw.circle(screen, WHITE, (int(
                column*SQUARESIZE+SQUARESIZE/2), int(row*SQUARESIZE+SQUARESIZE+SQUARESIZE/2)), RADIUS)  # Create the white slots for inserting the coins

    for column in range(COLUMN_COUNT):
        for row in range(ROW_COUNT):
            if board[row][column] == PLAYER_COIN:
                pygame.draw.circle(screen, RED, (int(
                    column*SQUARESIZE+SQUARESIZE/2), height-int(row*SQUARESIZE+SQUARESIZE/2)), RADIUS)  # Create the red coins for the Player
            elif board[row][column] == AI_COIN:
                pygame.draw.circle(screen, YELLOW, (int(
                    column*SQUARESIZE+SQUARESIZE/2), height-int(row*SQUARESIZE+SQUARESIZE/2)), RADIUS)  # Create the white coins for the AI
    pygame.display.update()


board = make_board()
print_board(board)

pygame.init()  # Initialize Connect4 with Pygame

# Radius is necessary to draw the white circles on the board
RADIUS = int(SQUARESIZE/2 - 5)

screen = pygame.display.set_mode(screen_size)  # Set the size of the screen
draw_board(board)
pygame.display.update()

# Initialize a font for the message that appears to declare the winner
myfont = pygame.font.SysFont("monospace", 80)

pygame.mixer.init()
pygame.mixer.music.load("Wii Music - Background Music.mp3")
pygame.mixer.music.play(loops=-1)  # Start playing the background music

white = [255, 255, 255]

turn = random.randint(PLAYER, AI)

clock = pygame.time.Clock()
pygame.display.flip()

clock.tick(3000)

running = False  # Turn to 'True' if someone gets 4 coins in a row

while not running:  # Main Loop that runs as long as running is 'False'. When running becomes 'True', the loop stops
    for event in pygame.event.get():  # FOR LOOP, when the game is over, exit the system and pygame. Need to run the file again in the terminal to play again
        if event.type == KEYDOWN:
            if event.key == K_ESCAPE:
                running = True
        if event.type == pygame.QUIT:
            sys.exit()
            running = True

        if event.type == pygame.MOUSEMOTION:  # IF CONDITIONAL, when the player clicks with their mouse during their turn, if there is a spot available on this column, then the last spot available becomes red. Also as long as the player has their mouse on the top bar of the screen, the red coin will follow according to the mouse's movements.
            pygame.draw.rect(screen, BLACK, (0, 0, width, SQUARESIZE))
            posx = event.pos[0]
            if turn == PLAYER:
                pygame.draw.circle(
                    screen, RED, (posx, int(SQUARESIZE/2)), RADIUS)
        pygame.display.update()  # Update the game as long as it runs

        # Click on the mouse to drop the coins(each of them has their own unique position)
        if event.type == pygame.MOUSEBUTTONDOWN:
            # Once the message about who won the game appears, the red coin in the top bar disappears
            pygame.draw.rect(screen, BLACK, (0, 0, width, SQUARESIZE))

            if turn == PLAYER:
                posx = event.pos[0]
                col = int(math.floor(posx/SQUARESIZE))

                if valid_location(board, col):
                    row = empty_slot(board, col)
                    insert_coin(board, row, col, PLAYER_COIN)

                    if winning_row(board, PLAYER_COIN):
                        # The message is printed within the top bar when the game is over and the player wins
                        label = myfont.render("Player wins!!", 1, RED)
                        # We have created our game objects, we need to actually render them.
                        screen.blit(label, (40, 10))
                        running = True

                    turn = (turn+1) % 2
                    print_board(board)
                    draw_board(board)

    if turn == AI and not running:  # IF CONDITIONAL, if the game is not over, then the AI keeps playing until there is a winner
        col, minimax_score = minimax(board, 6, -math.inf, math.inf, True)

        if valid_location(board, col):
            row = empty_slot(board, col)
            insert_coin(board, row, col, AI_COIN)

            if winning_row(board, AI_COIN):
                # The message is printed within the top bar when the game is over and the AI wins
                label = myfont.render("AI wins!!", 1, YELLOW)
                # We have created our game objects, we need to actually render them.
                screen.blit(label, (40, 10))
                running = True

            print_board(board)
            draw_board(board)

            turn += 1
            turn = turn % 2

    if running == True:
        pygame.mixer.music.stop()
        pygame.mixer.quit()  # Strop the background music

    if running:  # Secret question for obtaining the URL
        background = input(
            "What is the initial color of the empty spots for the coins ?: ")
        if background == "white":
            screen.fill(white)
            pygame.display.update()
        pygame.time.wait(3000)

# Dictionary of options after the game is over
obj = {
    "First Option": "AI wins!!",
    "Second Option": "Player wins!!",
    "Third Option": "Draw!",
    "Next Round": "Wanna play again?"
}

with open("board.json", "w") as f:  # Convert obj into a json file
    json.dump(obj, f)

with open("board.json", "r") as f:  # Read obj into a json file
    thing = json.load(f)
print(thing)

# Open a YouTube video for Pygame Beginners
webbrowser.open('https://www.youtube.com/watch?v=FfWpgLFMI7w')
