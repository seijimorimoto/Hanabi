package com.fossgalaxy.games.fireworks.ai.HisGranAha;

import com.fossgalaxy.games.fireworks.ai.Agent;
import com.fossgalaxy.games.fireworks.ai.iggi.Utils;
import com.fossgalaxy.games.fireworks.ai.rule.logic.DeckUtils;
import com.fossgalaxy.games.fireworks.state.*;
import com.fossgalaxy.games.fireworks.state.actions.*;

import java.util.*;
import java.util.stream.Collectors;

/**
 * Agent that plays Hanabi using MCTS but replacing its rollout phase with the predictions obtained
 * via a neural network.
 */
public class HisGranAha implements Agent {
    // Constants.
    public static final double EXPLORATION_CONST = Math.sqrt(2);
    public static final String MODEL_CONFIG_PATH = "src/main/resources/hanabi_nn_new.json";
    public static final String MODEL_WEIGHTS_PATH = "src/main/resources/hanabi_nn_new.h5";
    public static final int NUM_ACTIONS = 60;
    public static final int TIME_LIMIT = 1000;

    // Attributes of the class.
    private NeuralNetwork nn;
    private Set<NNState> visitedStates;
    private Map<NNState, double[]> policies;
    private Map<NNState, double[]> qValues;
    private Map<NNState, int[]> frequencyOfActions;

    /**
     * Constructs an instance of this agent and imports the neural network from the file specified
     * in the constants.
     */
    public HisGranAha() {
        nn = new NeuralNetwork();
        try {
            nn.importKerasModel(MODEL_CONFIG_PATH, MODEL_WEIGHTS_PATH);
        } catch (Exception e) {
            System.err.println("Could not load the Neural Network Keras model");
        }
    }

    /**
     * Action that is triggered when is the turn of this agent to make a move in the game.
     * @param agentID The Id (position) of the agent within the current game.
     * @param state The current state of the game.
     * @return The action chosen by the agent.
     */
    @Override
    public Action doMove(int agentID, GameState state) {
        // Initialize variables.
        visitedStates = new HashSet<>();
        policies = new HashMap<>();
        qValues = new HashMap<>();
        frequencyOfActions = new HashMap<>();

        long finishTime = System.currentTimeMillis() + TIME_LIMIT;

        // Map each slot in the hand to the list of possible cards that could be in it.
        Map<Integer, List<Card>> possibleCards = DeckUtils.bindCard(agentID, state.getHand(agentID), state.getDeck().toList());

        // Order the slots according to the size of the list of possible cards. Slots with fewer possible cards will
        // appear first in the list.
        List<Integer> bindOrder = DeckUtils.bindOrder(possibleCards);

        // Perform the MCTS tree search as long as we haven't exceeded the time threshold.
        while (System.currentTimeMillis() < finishTime) {
            GameState stateCopy = state.getCopy();

            // Randomly choose one of the possible cards for each slot and assign it to them.
            // A card that is selected to be in one slot is guaranteed to not be chosen to be in another one.
            Map<Integer, Card> cardsInHand = DeckUtils.bindCards(bindOrder, possibleCards);

            Deck deck = stateCopy.getDeck();
            Hand myHand = stateCopy.getHand(agentID);

            // Iterate over all the slots and assign each selected possible card to the slots, but this time using the
            // Hand object, so as to modify the game state. Also, remove each selected possible card from the Deck.
            for (int slot = 0; slot < myHand.getSize(); slot++)
            {
                Card cardInHand = cardsInHand.get(slot);
                myHand.bindCard(slot, cardInHand);
                deck.remove(cardInHand);
            }
            deck.shuffle();

            // Perform an iteration of the MCTS algorithm.
            search(stateCopy, nn, agentID, agentID);
        }

        int agentOffset = getPlayerOffset(agentID, agentID, state.getPlayerCount());
        NNState nnState = new NNState(state, agentOffset);
        return getBestExploitationAction(nnState, agentID, state.getPlayerCount());
    }

    /**
     * Performs an iteration of the MCTS algorithm.
     * @param state The state currently being explored in the search tree (corresponds to a node).
     * @param nn The network that will be used to replace the rollout phase.
     * @param thisAgentId The Id of this HisGranAha agent.
     * @param nextAgentID The Id of the agent that can take an action from the current state being explored.
     * @return The resulting value of the current state (it is propagated backwards while the recursion unwinds).
     */
    protected double search(GameState state, NeuralNetwork nn, int thisAgentId, int nextAgentID) {
        // When on a terminal state, return the actual score of the game.
        if (state.isGameOver()) {
            return state.getScore();
        }

        // Create a NNState from the current state, i.e. simplified representation of the state understandable
        // by our neural network.
        int playerCount = state.getPlayerCount();
        int agentOffset = getPlayerOffset(thisAgentId, nextAgentID, playerCount);
        NNState nnState = new NNState(state, agentOffset);

        // If we are currently in a leaf node, add it to the set of visited states, calculate the policy and
        // value of that node using the neural network and returned the value.
        if (!visitedStates.contains(nnState)) {
            visitedStates.add(nnState);
            NeuralNetwork.NeuralNetworkOutput nnOutputs = nn.predict(nnState);
            policies.put(nnState, nnOutputs.policy);
            return nnOutputs.value;
        }

        double maxUCB = -Double.MAX_VALUE;
        int bestActionId = -1;

        // Get the legal actions that can be performed by the 'nextAgentID' given the current state.
        Collection<Integer> legalActionIds = getPlayerLegalMoves(state, nextAgentID)
                .stream()
                .map(action -> getActionId(action, thisAgentId, playerCount))
                .collect(Collectors.toCollection(ArrayList::new));

        // Iterate over all legal actions Dds and find the one that leads to the greatest UCB value.
        for (int legalActionId : legalActionIds) {
            double qValue = qValues.getOrDefault(nnState, new double[NUM_ACTIONS])[legalActionId];
            double policy = policies.getOrDefault(nnState, new double[NUM_ACTIONS])[legalActionId];
            int[] freqOfActionsInState = frequencyOfActions.getOrDefault(nnState, new int[NUM_ACTIONS]);
            int totalFreqOfActionsInState = Arrays.stream(freqOfActionsInState).sum();
            double actionFreq = freqOfActionsInState[legalActionId];
            double actionUCB = qValue + EXPLORATION_CONST * policy * Math.sqrt(totalFreqOfActionsInState) / (1 + actionFreq);
            if (actionUCB > maxUCB) {
                maxUCB = actionUCB;
                bestActionId = legalActionId;
            }
        }

        // Based on the action Id, retrieve the best action that can be taken from the current state
        // and apply it to a copy of the state, so as to keep the current one (needed for the rest of
        // method) and the new one (used as a parameter for the recursive call on search).
        Action bestAction = getAction(bestActionId, thisAgentId, playerCount);
        GameState nextState = state.getCopy();
        bestAction.apply(nextAgentID, nextState);

        // Obtain a value by calling search recursively (essentially, going deeper into the tree).
        double value = search(nextState, nn, thisAgentId, (nextAgentID + 1) % playerCount);

        // Update the qValues by considering the value that was returned (back-propagated) from the
        // recursive search call. Also update N(s,a) by 1, where s is the current state and a is the action
        // that was taken (i.e. the one that had the highest UCB value).
        double[] qValuesForState = qValues.getOrDefault(nnState, new double[NUM_ACTIONS]);
        int[] freqOfActionsForState = frequencyOfActions.getOrDefault(nnState, new int[NUM_ACTIONS]);
        int freqOfBestAction = freqOfActionsForState[bestActionId];
        qValuesForState[bestActionId] = (freqOfBestAction * qValuesForState[bestActionId] + value) / (freqOfBestAction + 1);
        freqOfActionsForState[bestActionId] = freqOfBestAction + 1;

        qValues.put(nnState, qValuesForState);
        frequencyOfActions.put(nnState, freqOfActionsForState);

        // Back-propagate the result obtained by calling search recursively.
        return value;
    }

    /**
     * Retrieves the legal moves that can be performed by an agent from a given state.
     * @param state The state from which legal moves are going to be calculated.
     * @param agentID The agent that is supposed to take an action from the given state.
     * @return The collection of legal actions that the agent can perform.
     */
    protected Collection<Action> getPlayerLegalMoves(GameState state, int agentID) {
        Collection<Action> allPossibleActions = Utils.generateAllActions(agentID, state.getPlayerCount());
        return allPossibleActions
                .stream()
                .filter(action -> action.isLegal(agentID, state))
                .collect(Collectors.toList());
    }

    /**
     * Retrieves the Id of a particular action.
     * @param action The action whose Id is desired.
     * @param thisAgentId The Id of this HisGranAha agent.
     * @param playerCount The number of current players in the game.
     * @return The Id of the desired action.
     */
    public int getActionId(Action action, int thisAgentId, int playerCount) {
        if (action instanceof DiscardCard)
            return ((DiscardCard) action).slot;
        if (action instanceof PlayCard)
            return 5 + ((PlayCard) action).slot;

        int actionId = 0;
        int playerToldId = 0;
        if (action instanceof TellColour) {
            actionId = 10 + ((TellColour) action).colour.ordinal();
            playerToldId = ((TellColour) action).player;
        }
        else if (action instanceof TellValue) {
            actionId = 15 + ((TellValue) action).value - 1;
            playerToldId = ((TellValue) action).player;
        }

        int playerOffset = getPlayerOffset(thisAgentId, playerToldId, playerCount);
        actionId = actionId + 10 * playerOffset;
        return actionId;
    }

    /**
     * Retrieves an action based on a given Id.
     * @param actionId The Id of the action to be retrieved.
     * @param thisAgentId The Id of this HisGranAha agent.
     * @param playerCount The number of current players in the game.
     * @return The action corresponding to the given Id.
     */
    public Action getAction(int actionId, int thisAgentId, int playerCount) {
        if (actionId < 5)
            return new DiscardCard(actionId);
        if (actionId < 10)
            return new PlayCard(actionId - 5);

        int offset = actionId / 10 - 1;
        int playerToTellId = (thisAgentId + offset) % playerCount;

        if (actionId < 15 || (actionId >= 20 && actionId < 25) || (actionId >= 30 && actionId < 35) ||
                (actionId >= 40 && actionId < 45) || (actionId >= 50 && actionId < 55)) {
            int colourToTellId = actionId % 10;
            CardColour colourToTell = null;
            switch (colourToTellId) {
                case 0: { colourToTell = CardColour.RED; break; }
                case 1: { colourToTell = CardColour.BLUE; break; }
                case 2: { colourToTell = CardColour.GREEN; break; }
                case 3: { colourToTell = CardColour.ORANGE; break; }
                case 4: { colourToTell = CardColour.WHITE; break; }
            }
            return new TellColour(playerToTellId, colourToTell);
        }

        int valueToTell = actionId % 10 - 4;
        return new TellValue(playerToTellId, valueToTell);
    }

    /**
     * Returns the action that leads to the maximum expected value from a given state.
     * @param nnState The state from which the best action is going to be selected.
     * @param thisAgentId The Id of this HisGranAha agent.
     * @param playerCount The number of current players in the game.
     * @return The best action according to an exploitation mindset.
     */
    public Action getBestExploitationAction(NNState nnState, int thisAgentId, int playerCount) {
        double[] qValuesForState = qValues.get(nnState);
        double bestQValue = -Double.MAX_VALUE;
        int bestActionId = -1;

        // Iterates over all Q(s,a) and finds the index of the maximum.
        for (int i = 0; i < qValuesForState.length; i++) {
            if (qValuesForState[i] > bestQValue) {
                bestQValue = qValuesForState[i];
                bestActionId = i;
            }
        }

        // Returns the best action.
        return getAction(bestActionId, thisAgentId, playerCount);
    }

    /**
     * Returns the position of a player with respect to this HisGranAha position (Id) in the game.
     * @param thisAgentId The Id of this HisGranAha agent.
     * @param playerId The Id of the player whose offset from this agent is desired.
     * @param playerCount The number of current players in the game.
     * @return The offset of the player with respect to this agent.
     */
    public int getPlayerOffset(int thisAgentId, int playerId, int playerCount) {
        if (thisAgentId <= playerId)
            return playerId - thisAgentId;
        else
            return playerCount - thisAgentId + playerId;
    }
}
