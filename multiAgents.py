# multiAgents.py
# --------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent

class ReflexAgent(Agent):
	"""
	A reflex agent chooses an action at each choice point by examining
	its alternatives via a state evaluation function.

	The code below is provided as a guide.  You are welcome to change
	it in any way you see fit, so long as you don't touch our method
	headers.
	"""


	def getAction(self, gameState):
		"""
		You do not need to change this method, but you're welcome to.

		getAction chooses among the best options according to the evaluation function.

		Just like in the previous project, getAction takes a GameState and returns
		some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
		"""
		# Collect legal moves and child states
		legalMoves = gameState.getLegalActions()

		# Choose one of the best actions
		scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
		bestScore = max(scores)
		bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
		chosenIndex = random.choice(bestIndices) # Pick randomly among the best

		"Add more of your code here if you want to"

		return legalMoves[chosenIndex]

	def evaluationFunction(self, currentGameState, action):
		"""
		Design a better evaluation function here.

		The evaluation function takes in the current and proposed child
		GameStates (pacman.py) and returns a number, where higher numbers are better.

		The code below extracts some useful information from the state, like the
		remaining food (newFood) and Pacman position after moving (newPos).
		newScaredTimes holds the number of moves that each ghost will remain
		scared because of Pacman having eaten a power pellet.

		Print out these variables to see what you're getting, then combine them
		to create a masterful evaluation function.
		"""
		# Useful information you can extract from a GameState (pacman.py)
		childGameState = currentGameState.getPacmanNextState(action)
		newPos = childGameState.getPacmanPosition()
		newFood = childGameState.getFood()
		newGhostStates = childGameState.getGhostStates()
		newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

		currFood = currentGameState.getFood()
		newPellets = childGameState.getCapsules()
		currPellets = currentGameState.getCapsules()
		closestGhost = newGhostStates[0]
		closestFoodPos = None

		if action == "Stop":
			return -1

		for ghostState in newGhostStates:
			if manhattanDistance(ghostState.getPosition(), newPos) < manhattanDistance(closestGhost.getPosition(), newPos):
				closestGhost = ghostState

		if closestGhost.getPosition() == newPos:
			return -9999999

		if len(newFood.asList()) == 0 and len(newPellets) == 0:
			return 9999999999
		elif len(newFood.asList()) > 0:
			finalScore = 0
			closestFoodPos = newFood.asList()[0]

			for foodPos in newFood.asList():
				if manhattanDistance(foodPos, newPos) < manhattanDistance(closestFoodPos, newPos):
					closestFoodPos = foodPos

			if len(newFood.asList()) < len(currFood.asList()) or len(newPellets) < len(currPellets):
				finalScore += 100
			else:
				finalScore += (5 / manhattanDistance(newPos, closestFoodPos))
		elif len(newPellets) > 0:
			finalScore = 0
			closestPelletPos = newPellets[0]

			for pelletPos in newPellets:
				if manhattanDistance(pelletPos, newPos) < manhattanDistance(closestPelletPos, newPos):
					closestPelletPos = pelletPos

			if len(newFood.asList()) < len(currFood.asList()) or len(newPellets) < len(currPellets):
				finalScore += 100
			else:
				finalScore += (5 / manhattanDistance(newPos, closestPelletPos))

		if closestGhost.scaredTimer == 0:
				finalScore += 0.5 * (manhattanDistance(newPos, closestGhost.getPosition()) ** 0.5)

		return finalScore

def scoreEvaluationFunction(currentGameState):
	"""
	This default evaluation function just returns the score of the state.
	The score is the same one displayed in the Pacman GUI.

	This evaluation function is meant for use with adversarial search agents
	(not reflex agents).
	"""
	return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
	"""
	This class provides some common elements to all of your
	multi-agent searchers.  Any methods defined here will be available
	to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

	You *do not* need to make any changes here, but you can if you want to
	add functionality to all your adversarial search agents.  Please do not
	remove anything, however.

	Note: this is an abstract class: one that should not be instantiated.  It's
	only partially specified, and designed to be extended.  Agent (game.py)
	is another abstract class.
	"""

	def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
		self.index = 0 # Pacman is always agent index 0
		self.evaluationFunction = util.lookup(evalFn, globals())
		self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
	"""
	Your minimax agent (question 2)
	"""

	def getAction(self, gameState):
		"""
		Returns the minimax action from the current gameState using self.depth
		and self.evaluationFunction.

		Here are some method calls that might be useful when implementing minimax.

		gameState.getLegalActions(agentIndex):
		Returns a list of legal actions for an agent
		agentIndex=0 means Pacman, ghosts are >= 1

		gameState.getNextState(agentIndex, action):
		Returns the child game state after an agent takes an action

		gameState.getNumAgents():
		Returns the total number of agents in the game

		gameState.isWin():
		Returns whether or not the game state is a winning state

		gameState.isLose():
		Returns whether or not the game state is a losing state
		"""

		def minimaxValue(agent, currentState, currdepth):
			if currdepth >= self.depth or currentState.isWin() or currentState.isLose() or len(currentState.getLegalActions(agent)) == 0:
				return (self.evaluationFunction(currentState), None)
			
			if agent == 0:
				maxVal = float('-inf')
				maxAct = None

				for act in currentState.getLegalActions(agent):
					(currVal, currAct) = minimaxValue(agent + 1, currentState.getNextState(agent, act), currdepth)

					if currVal > maxVal:
						maxVal = currVal
						maxAct = act

				return (maxVal, maxAct)

			elif agent > 0 and agent < gameState.getNumAgents() - 1:
				minVal = float('inf')
				minAct = None

				for act in currentState.getLegalActions(agent):
					(currVal, currAct) = minimaxValue(agent + 1, currentState.getNextState(agent, act), currdepth)

					if currVal < minVal:
						minVal = currVal
						minAct = currAct

				return (minVal, minAct)

			else:
				minVal = float('inf')
				minAct = None

				for act in currentState.getLegalActions(agent):
					(currVal, currAct) = minimaxValue(0, currentState.getNextState(agent, act), currdepth + 1)

					if currVal < minVal:
						minVal = currVal
						minAct = currAct

				return (minVal, minAct)

		return minimaxValue(0, gameState, 0)[1]

class AlphaBetaAgent(MultiAgentSearchAgent):
	"""
	Your minimax agent with alpha-beta pruning (question 3)
	"""

	def getAction(self, gameState):
		"""
		Returns the minimax action using self.depth and self.evaluationFunction
		"""
		def AB(agent, currentState, currdepth, atMost, atLeast):
			if currdepth >= self.depth or currentState.isWin() or currentState.isLose() or len(currentState.getLegalActions(agent)) == 0:
				return (self.evaluationFunction(currentState), None)
			
			if agent == 0:
				maxVal = float('-inf')
				maxAct = None

				for act in currentState.getLegalActions(agent):
					(currVal, currAct) = AB(agent + 1, currentState.getNextState(agent, act), currdepth, atMost, atLeast)

					if currVal > maxVal:
						maxVal = currVal
						maxAct = act

					if maxVal > atMost:
						return (maxVal, maxAct)

					if maxVal > atLeast:
						atLeast = maxVal

				return (maxVal, maxAct)

			elif agent > 0 and agent < gameState.getNumAgents() - 1:
				minVal = float('inf')
				minAct = None

				for act in currentState.getLegalActions(agent):
					(currVal, currAct) = AB(agent + 1, currentState.getNextState(agent, act), currdepth, atMost, atLeast)

					if currVal < minVal:
						minVal = currVal
						minAct = currAct

					if minVal < atLeast:
						return (minVal, minAct)

					if minVal < atMost:
						atMost = minVal

				return (minVal, minAct)

			else:
				minVal = float('inf')
				minAct = None

				for act in currentState.getLegalActions(agent):
					(currVal, currAct) = AB(0, currentState.getNextState(agent, act), currdepth + 1, atMost, atLeast)

					if currVal < minVal:
						minVal = currVal
						minAct = currAct

					if minVal < atLeast:
						return (minVal, minAct)

					if minVal < atMost:
						atMost = minVal

				return (minVal, minAct)

		return AB(0, gameState, 0, float('inf'), float('-inf'))[1]

class ExpectimaxAgent(MultiAgentSearchAgent):
	"""
	  Your expectimax agent (question 4)
	"""

	def getAction(self, gameState):
		"""
		Returns the expectimax action using self.depth and self.evaluationFunction

		All ghosts should be modeled as choosing uniformly at random from their
		legal moves.
		"""
		def expectimax(agent, currentState, currdepth):
			if currdepth >= self.depth or currentState.isWin() or currentState.isLose() or len(currentState.getLegalActions(agent)) == 0:
				return (self.evaluationFunction(currentState), None)
			
			if agent == 0:
				maxVal = float('-inf')
				maxAct = None

				for act in currentState.getLegalActions(agent):
					(currVal, currAct) = expectimax(agent + 1, currentState.getNextState(agent, act), currdepth)

					if currVal > maxVal:
						maxVal = currVal
						maxAct = act

				return (maxVal, maxAct)

			elif agent > 0 and agent < gameState.getNumAgents() - 1:
				expectation = 0

				for act in currentState.getLegalActions(agent):
					(currVal, currAct) = expectimax(agent + 1, currentState.getNextState(agent, act), currdepth)

					expectation += (1 / len(currentState.getLegalActions(agent))) * currVal

				return (expectation, currentState.getLegalActions(agent)[random.randint(0, len(currentState.getLegalActions(agent)) - 1)])

			else:
				expectation = 0

				for act in currentState.getLegalActions(agent):
					(currVal, currAct) = expectimax(0, currentState.getNextState(agent, act), currdepth + 1)

					expectation += (1 / len(currentState.getLegalActions(agent))) * currVal

				return (expectation, currentState.getLegalActions(agent)[random.randint(0, len(currentState.getLegalActions(agent)) - 1)])

		return expectimax(0, gameState, 0)[1]

def betterEvaluationFunction(currentGameState):
	"""
	Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
	evaluation function (question 5).

	DESCRIPTION: <write something here so we know what you did>

		gameState.getLegalActions(agentIndex):
		Returns a list of legal actions for an agent
		agentIndex=0 means Pacman, ghosts are >= 1

		gameState.getNextState(agentIndex, action):
		Returns the child game state after an agent takes an action

		gameState.getNumAgents():
		Returns the total number of agents in the game

		gameState.isWin():
		Returns whether or not the game state is a winning state

		gameState.isLose():
		Returns whether or not the game state is a losing state

	"""

	pacPos = currentGameState.getPacmanPosition()
	ghostStates = currentGameState.getGhostStates()
	scaredTimes = [ghostState.scaredTimer for ghostState in ghostStates]
	foodList = currentGameState.getFood().asList()
	pelletList = currentGameState.getCapsules()

	closestGhost = ghostStates[0]
	closestFoodPos = None
	finalScore = 0

	if len(foodList) == 0:
		return 999999999999
	if len(pelletList) == 0:
		finalScore += 100

	closestFoodPos = foodList[0]

	"""
	for foodPos in foodList:
		if manhattanDistance(foodPos, pacPos) < manhattanDistance(closestFoodPos, pacPos):
			closestFoodPos = foodPos

	finalScore += (2 / manhattanDistance(pacPos, closestFoodPos))

	"""
	for foodPos in foodList:
		finalScore += (1 / manhattanDistance(pacPos, foodPos)) + (25 / len(foodList))

	finalScore += (10 / max(1, len(foodList)))
	finalScore += (100 / max(1, len(pelletList)))

	for ghostState in ghostStates:
		if manhattanDistance(ghostState.getPosition(), pacPos) < manhattanDistance(closestGhost.getPosition(), pacPos):
			closestGhost = ghostState

	if closestGhost.scaredTimer == 0:
		finalScore += (manhattanDistance(pacPos, closestGhost.getPosition()) ** 0.5)

	finalScore += currentGameState.getScore()

	return finalScore

# Abbreviation
better = betterEvaluationFunction
