package com.fossgalaxy.games.fireworks.ai.HisGranAha;

import com.fossgalaxy.games.fireworks.ai.Agent;
import com.fossgalaxy.games.fireworks.ai.iggi.Utils;
import com.fossgalaxy.games.fireworks.ai.rule.logic.DeckUtils;
import com.fossgalaxy.games.fireworks.state.*;
import com.fossgalaxy.games.fireworks.state.actions.*;
import javafx.util.Pair;

import java.util.*;
import java.util.stream.Collectors;

public class HisGranAha implements Agent {
    public static final double EXPLORATION_CONST = Math.sqrt(2);
    public static final int NUM_ACTIONS = 60;
    public static final int TIME_LIMIT = 1000;

    private NeuralNetwork nn;
    private Set<GameState> visitedStates;
    private Map<GameState, double[]> policies;
    private Map<GameState, double[]> qValues;
    private Map<GameState, int[]> frequencyOfActions;

    public HisGranAha() {
        nn = new NeuralNetwork();
    }

    @Override
    public Action doMove(int agentID, GameState state) {
        visitedStates = new HashSet<>();
        policies = new HashMap<>();
        qValues = new HashMap<>();
        frequencyOfActions = new HashMap<>();

        Set<GameState> initialStates = new HashSet<>();
        long finishTime = System.currentTimeMillis() + TIME_LIMIT;

        // Map each slot in the hand to the list of possible cards that could be in it.
        Map<Integer, List<Card>> possibleCards = DeckUtils.bindCard(agentID, state.getHand(agentID), state.getDeck().toList());

        // Order the slots according to the size of the list of possible cards. Slots with fewer possible cards will
        // appear first in the list.
        List<Integer> bindOrder = DeckUtils.bindOrder(possibleCards);

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

            initialStates.add(stateCopy);
//            search(stateCopy, nn, agentID, agentID);
        }

        return getBestExploitationAction(initialStates, agentID, state.getPlayerCount());
    }

    protected double search(GameState state, NeuralNetwork nn, int thisAgentId, int nextAgentID) {
        if (state.isGameOver()) {
            return state.getScore();
        }

        int playerCount = state.getPlayerCount();

        if (!visitedStates.contains(state)) {
            visitedStates.add(state);
            int agentOffset = getPlayerOffset(thisAgentId, nextAgentID, playerCount);
            NNState nnState = new NNState(state, agentOffset);
            NeuralNetwork.NeuralNetworkOutput nnOutputs = nn.predict(nnState);
            policies.put(state, nnOutputs.policy);
            return nnOutputs.value;
        }

        double maxUCB = -Double.MAX_VALUE;
        int bestActionId = -1;

        Collection<Integer> legalActionIds = getPlayerLegalMoves(state, nextAgentID)
                .stream()
                .map(action -> getActionId(action, thisAgentId, playerCount))
                .collect(Collectors.toCollection(ArrayList::new));

        for (int legalActionId : legalActionIds) {
            double qValue = qValues.getOrDefault(state, new double[NUM_ACTIONS])[legalActionId];
            double policy = policies.getOrDefault(state, new double[NUM_ACTIONS])[legalActionId];
            int[] freqOfActionsInState = frequencyOfActions.getOrDefault(state, new int[NUM_ACTIONS]);
            int totalFreqOfActionsInState = Arrays.stream(freqOfActionsInState).sum();
            double actionFreq = freqOfActionsInState[legalActionId];
            double actionUCB = qValue + EXPLORATION_CONST * policy * Math.sqrt(totalFreqOfActionsInState) / (1 + actionFreq);
            if (actionUCB > maxUCB) {
                maxUCB = actionUCB;
                bestActionId = legalActionId;
            }
        }

        Action bestAction = getAction(bestActionId, thisAgentId, playerCount);
        GameState nextState = state.getCopy();
        bestAction.apply(nextAgentID, nextState);
        double value = search(state, nn, thisAgentId, (nextAgentID + 1) % playerCount);

        double[] qValuesForState = qValues.get(state);
        int freqOfBestAction = frequencyOfActions.get(state)[bestActionId];
        qValuesForState[bestActionId] = (freqOfBestAction * qValuesForState[bestActionId] + value) / (freqOfBestAction + 1);
        frequencyOfActions.get(state)[bestActionId]++;

        return value;
    }

    protected Collection<Action> getPlayerLegalMoves(GameState state, int agentID) {
        Collection<Action> allPossibleActions = Utils.generateAllActions(agentID, state.getPlayerCount());
        return allPossibleActions.stream().filter(action -> action.isLegal(agentID, state)).collect(Collectors.toList());
    }

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

    public Action getBestExploitationAction(Set<GameState> initialStates, int thisAgentId, int playerCount) {
        double bestQValue = -Double.MAX_VALUE;
        Action bestAction = null;
        for (GameState state : initialStates) {
            Pair<Action, Double> actionQValuePair = getBestExploitationAction(state, thisAgentId, playerCount);
            if (actionQValuePair.getValue() > bestQValue) {
                bestQValue = actionQValuePair.getValue();
                bestAction = actionQValuePair.getKey();
            }
        }
        return bestAction;
    }

    public Pair<Action, Double> getBestExploitationAction(GameState state, int thisAgentId, int playerCount) {
        double[] qValuesForState = qValues.get(state);
        double bestQValue = -Double.MAX_VALUE;
        int bestActionId = -1;

        for (int i = 0; i < qValuesForState.length; i++) {
            if (qValuesForState[i] > bestQValue) {
                bestQValue = qValuesForState[i];
                bestActionId = i;
            }
        }

        Action bestAction = getAction(bestActionId, thisAgentId, playerCount);
        return new Pair<>(bestAction, bestQValue);
    }

    public int getPlayerOffset(int thisAgentId, int playerId, int playerCount) {
        if (thisAgentId <= playerId)
            return playerId - thisAgentId;
        else
            return playerCount - thisAgentId + playerId;
    }
}
