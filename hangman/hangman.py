# Hangman game
#

# -----------------------------------
# Helper code
# You don't need to understand this helper code,
# but you will have to know how to use the functions
# (so be sure to read the docstrings!)

import random

wordlist = []
with open('words2.txt','r') as f:
    for line in f:
        for word in line.split():
             wordlist.append(word)
wlcopy = wordlist[:]     
for e in wlcopy:
    if len(e) < 7 or len(e) > 10:
        wordlist.remove(e)
        
for e in wordlist:
    for i in e:
        if i == "ü" or i == "ä" or i == "ö":
            wordlist.remove(e)
            break

"""
WORDLIST_FILENAME = "words.txt"

def loadWords():
    ""
    Returns a list of valid words. Words are strings of lowercase letters.
    
    Depending on the size of the word list, this function may
    take a while to finish.
    ""
    print("Loading word list from file...")
    # inFile: file
    inFile = open(WORDLIST_FILENAME, 'r')
    # line: string
    line = inFile.readline()
    # wordlist: list of strings
    wordlist = line.split()
    print("  ", len(wordlist), "words loaded.")
    return wordlist
"""

def chooseWord(wordlist):
    """
    wordlist (list): list of words (strings)

    Returns a word from wordlist at random
    """
    return random.choice(wordlist)

# end of helper code
# -----------------------------------

# Load the list of words into the variable wordlist
# so that it can be accessed from anywhere in the program
# wordlist = loadWords()

def isWordGuessed(secretWord, lettersGuessed):
    '''
    secretWord: string, the word the user is guessing
    lettersGuessed: list, what letters have been guessed so far
    returns: boolean, True if all the letters of secretWord are in lettersGuessed;
      False otherwise
    '''
    # FILL IN YOUR CODE HERE...
    F = 0
    for e in secretWord:
        if e not in lettersGuessed:
            F += 1
    if F == 0:
        return(True)
    else:
        return(False)


def getGuessedWord(secretWord, lettersGuessed):
    '''
    secretWord: string, the word the user is guessing
    lettersGuessed: list, what letters have been guessed so far
    returns: string, comprised of letters and underscores that represents
      what letters in secretWord have been guessed so far.
    '''
    # FILL IN YOUR CODE HERE...
    curr = ""
    for e in secretWord:
        if e in lettersGuessed:
            curr += e
        else:
            curr += "_ "
    
    return(curr)


def getAvailableLetters(lettersGuessed):
    '''
    lettersGuessed: list, what letters have been guessed so far
    returns: string, comprised of letters that represents what letters have not
      yet been guessed.
    '''
    # FILL IN YOUR CODE HERE...
    import string
    ans = ""
    for e in string.ascii_lowercase:
        if e not in lettersGuessed:
            ans += e
    return(ans)
    
    

# def hangman():
    '''
    secretWord: string, the secret word to guess.

    Starts up an interactive game of Hangman.

    * At the start of the game, let the user know how many 
      letters the secretWord contains.

    * Ask the user to supply one guess (i.e. letter) per round.

    * The user should receive feedback immediately after each guess 
      about whether their guess appears in the computers word.

    * After each round, you should also display to the user the 
      partially guessed word so far, as well as letters that the 
      user has not yet guessed.

    Follows the other limitations detailed in the problem write-up.
    '''
    # FILL IN YOUR CODE HERE...
    
import string
import time
secretWord = chooseWord(wordlist)
print("Welcome to the Game, Hangman!")
# print(secretWord)
lettersGuessed = ""
print("I am thinking of a word that is", len(secretWord), "letters long")
rounds = int(input("How many rounds? "))
while rounds not in list(range(20)):
    rounds = input("How many rounds? ")
while rounds > 0:
    print("You have ", rounds, "guesses left")
    print("Available letters: ",getAvailableLetters(lettersGuessed))
    print("----------------------------")
    guess = input("Please guess a letter : ")
    while guess not in string.ascii_lowercase or len(guess) > 1:
        guess = input("Please guess a lower case letter: ")
        # guess = guess.lower()
    while guess in lettersGuessed:
        guess = input("Uups, you already guessed that letter. Please guess another letter: ")
    lettersGuessed = lettersGuessed + guess
    print("Evaluating...")
    time.sleep(1)
    if guess in secretWord:
        print("Good guess: ", getGuessedWord(secretWord,lettersGuessed))
    else:
        print("Bad luck! This letters is not in the secret Word: ", getGuessedWord(secretWord,lettersGuessed))
        rounds -= 1
    if isWordGuessed(secretWord,lettersGuessed) == True:
        print("Congratulations Baby ;), you guessed the word!")
        print(secretWord)
        break
    print("")
if isWordGuessed(secretWord,lettersGuessed) == True:
    print()
else:
    print("You ran out of guesses ... man hanged!")
    print("The secret word was : ", secretWord)
        
    
    
    
    



# When you've completed your hangman function, uncomment these two lines
# and run this file to test! (hint: you might want to pick your own
# secretWord while you're testing)

# secretWord = chooseWord(wordlist).lower()
# hangman(secretWord)
