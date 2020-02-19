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
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()
        if 'Stop' in legalMoves:
            legalMoves.remove('Stop')
        numFood = len(gameState.getFood().asList())
        ghoststates = gameState.getGhostStates()
        ghost = ghoststates[0]
        timer = ghost.scaredTimer
        if timer > 5:
            newPosits = [gameState.generatePacmanSuccessor(action).getPacmanPosition() for action in legalMoves]
            scores = [manhattanDistance(pos, ghost.getPosition()) for pos in newPosits]
            bestScore = min(scores)
            bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
            for ind in bestIndices:
                if legalMoves[ind] == ghost.getDirection():
                    return legalMoves[ind]
            return legalMoves[random.choice(bestIndices)]



        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        if numFood > 10:
            for ind in bestIndices:
                if (legalMoves[ind] == 'South') or (legalMoves[ind] == 'West'):
                    return legalMoves[ind]

        chosenIndex = random.choice(bestIndices) # Pick randomly among the best





        return legalMoves[chosenIndex]


    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        from util import manhattanDistance

        def closestFood( position, foodList ):
            if len(foodList) == 0:
                return 0
            foodDistances = []
            for food in foodList:
                foodDistances.append( manhattanDistance( position, food) )
            nearestFoodDist = min(foodDistances)
            bestIndices = [index for index in range(len(foodDistances)) if foodDistances[index] == nearestFoodDist]
            foodClosest = [foodList[index] for index in bestIndices]


            return min(foodClosest)


        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newFoodList = newFood.asList()
        if len(newFoodList) == 0:
            return 10000
        curFood = currentGameState.getFood()
        curFoodList = curFood.asList()
        newGhostStates = successorGameState.getGhostStates()
        for ghost in newGhostStates:
            newGhostPos = ghost.getPosition()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]
        scoreCalc = 0
        powerCapsul = successorGameState.getCapsules()
        if len(powerCapsul) == 0:
            powerCapsul.append(min(newFoodList))
        distToGhost = manhattanDistance(newPos, newGhostPos)


        distToPower = manhattanDistance(newPos, powerCapsul[0])


        if distToGhost < 3:
            return 0
        if distToPower == 0:
            return 1000
        if newPos == powerCapsul[0]:
            return 1000

        closestFood = closestFood(newPos, curFoodList) #coord tup
        distToFood = manhattanDistance(newPos, closestFood)
        if distToFood == 0:
            scoreCalc += 500
            scoreCalc -= (75 / distToGhost)
            scoreCalc += (50 / distToPower)
            scoreCalc += successorGameState.getScore()
            return scoreCalc

        scoreCalc += (50 / distToPower)
        scoreCalc += (100 / distToFood)
        scoreCalc -= (75 / distToGhost)
        scoreCalc += successorGameState.getScore()
        return scoreCalc


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

        gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        def Minimaxer(sucGameState, depth, curAgent, numGhosts):
            if sucGameState.isWin() or sucGameState.isLose():
                return self.evaluationFunction(sucGameState)
            if depth == 0:
                return self.evaluationFunction(sucGameState)
            stateValue = 9999
            curDepthCurAgentActions = sucGameState.getLegalActions(curAgent)
            if curAgent == numGhosts:
                for curAction in curDepthCurAgentActions:
                    stateValue = min(stateValue, miniMaxer(sucGameState.generateSuccessor(curAgent,curAction), depth-1, numGhosts) )
            else:
                for curAction in curDepthCurAgentActions:
                    stateValue = min(stateValue, Minimaxer(sucGameState.generateSuccessor(curAgent,curAction), depth, curAgent + 1, numGhosts))
            return stateValue

        def miniMaxer(sucGameState, depth, numGhosts):
            if sucGameState.isWin() or sucGameState.isLose():
                return self.evaluationFunction(sucGameState)
            if depth == 0:
                return self.evaluationFunction(sucGameState)
            stateValue = -9999
            curDepthCurAgentActions = sucGameState.getLegalActions(0)
            for curAction in curDepthCurAgentActions:
                stateValue = max(stateValue, Minimaxer(sucGameState.generateSuccessor(0,curAction), depth, 1, numGhosts ))
            return stateValue

        numAgents = gameState.getNumAgents()
        numGhosts = numAgents - 1
        pacmanLegalActions = gameState.getLegalActions()

        bestScore = -9999
        optimalAction = Directions.STOP


        for action in pacmanLegalActions:
            prevActionScore = bestScore
            bestScore = max(prevActionScore, Minimaxer(gameState.generateSuccessor(0, action), self.depth, 1, numGhosts))
            if bestScore > prevActionScore:
                optimalAction = action
        return optimalAction

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        def Minimaxer(sucGameState, depth, curAgent, alpha, beta):
            if sucGameState.isWin() or sucGameState.isLose():
                return self.evaluationFunction(sucGameState)
            if depth == 0:
                return self.evaluationFunction(sucGameState)
            stateValue = 9999
            curDepthCurAgentActions = sucGameState.getLegalActions(curAgent)
            if curAgent == sucGameState.getNumAgents()-1:
                for curAction in curDepthCurAgentActions:
                    stateValue = min(stateValue, miniMaxer(sucGameState.generateSuccessor(curAgent,curAction), depth-1,  alpha, beta) )
                    if alpha > stateValue:
                        return stateValue
                    beta = min(beta, stateValue)
            else:
                for curAction in curDepthCurAgentActions:
                    stateValue = min(stateValue, Minimaxer(sucGameState.generateSuccessor(curAgent,curAction), depth, curAgent + 1,  alpha, beta))
                    if alpha > stateValue:
                        return stateValue
                    beta = min(beta, stateValue)
            return stateValue

        def miniMaxer(sucGameState, depth, alpha, beta):
            if sucGameState.isWin() or sucGameState.isLose():
                return self.evaluationFunction(sucGameState)
            if depth == 0:
                return self.evaluationFunction(sucGameState)
            stateValue = -9999
            curDepthCurAgentActions = sucGameState.getLegalActions(0)
            for curAction in curDepthCurAgentActions:
                stateValue = max(stateValue, Minimaxer(sucGameState.generateSuccessor(0,curAction), depth, 1,  alpha, beta ))
                if beta < stateValue:
                    return stateValue
                alpha = max(stateValue, alpha)
            return stateValue

        numAgents = gameState.getNumAgents()
        numGhosts = numAgents - 1
        pacmanLegalActions = gameState.getLegalActions()

        bestScore = -9999
        optimalAction = Directions.STOP
        alpha = -9999
        beta = 9999


        for action in pacmanLegalActions:
            prevActionScore = bestScore
            bestScore = max(prevActionScore, Minimaxer(gameState.generateSuccessor(0, action), self.depth, 1,  alpha, beta))
            if bestScore > prevActionScore:
                prevActionScore = bestScore
                optimalAction = action
            if beta < bestScore:
                return bestScore
            alpha = max(bestScore, alpha)
        return optimalAction


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


        def miniMaxer(sucGameState, depth):
            if sucGameState.isWin() or sucGameState.isLose():
                return self.evaluationFunction(sucGameState)
            if depth == 0:
                return self.evaluationFunction(sucGameState)
            stateValue = -9999
            curDepthCurAgentActions = sucGameState.getLegalActions(0)
            for curAction in curDepthCurAgentActions:
                stateValue = max(stateValue, MinimaxerEx(sucGameState.generateSuccessor(0,curAction), depth, 1))
            return stateValue

        def MinimaxerEx(sucGameState, depth, curAgent):
            if sucGameState.isWin() or sucGameState.isLose():
                return self.evaluationFunction(sucGameState)
            if depth == 0:
                return self.evaluationFunction(sucGameState)
            stateValue = 9999
            curDepthCurAgentActions = sucGameState.getLegalActions(curAgent)
            expected = 0
            if curAgent == sucGameState.getNumAgents()-1:
                for curAction in curDepthCurAgentActions:
                    expected += miniMaxer(sucGameState.generateSuccessor(curAgent,curAction), depth-1)
            else:
                for curAction in curDepthCurAgentActions:
                    expected +=  MinimaxerEx(sucGameState.generateSuccessor(curAgent,curAction), depth, curAgent + 1)
            return expected / len(curDepthCurAgentActions)


        pacmanLegalActions = gameState.getLegalActions()

        bestScore = -9999
        optimalAction = Directions.STOP


        for action in pacmanLegalActions:
            prevActionScore = bestScore
            bestScore = max(bestScore, MinimaxerEx(gameState.generateSuccessor(0, action), self.depth, 1))
            if bestScore > prevActionScore:
                optimalAction = action
        return optimalAction


def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
         evaluation function (question 5).
    """
    from util import manhattanDistance
    next_pacmanPos = currentGameState.getPacmanPosition()
    next_foodPos = [food for food in currentGameState.getFood().asList() if food]
    next_ghostsPos = currentGameState.getGhostStates()
    next_ghostsTimers = [ghostPos.scaredTimer for ghostPos in next_ghostsPos]
    ghostDist = min(manhattanDistance(next_pacmanPos, ghost.configuration.pos) for ghost in next_ghostsPos)
    closest_foodDist = min(manhattanDistance(next_pacmanPos, nextFood) for nextFood in next_foodPos) if next_foodPos else 0
    scared_time = min(next_ghostsTimers)
    foodLeft = -len(next_foodPos)
    ghostDistFeature = -2 / (ghostDist + 1) if scared_time == 0 else 0.5 / (ghostDist + 1)
    foodFeature = 0.4 / (closest_foodDist + 1)
    power_pelletsFeature = scared_time * 0.5
    game_score = currentGameState.getScore() * 0.6

    return foodLeft + ghostDistFeature + foodFeature + power_pelletsFeature + game_score


# Abbreviation
better = betterEvaluationFunction
