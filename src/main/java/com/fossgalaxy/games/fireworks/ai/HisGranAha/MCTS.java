package com.fossgalaxy.games.fireworks.ai.HisGranAha;

import com.fossgalaxy.games.fireworks.ai.Agent;
import com.fossgalaxy.games.fireworks.ai.iggi.Utils;
import com.fossgalaxy.games.fireworks.ai.mcts.IterationObject;
import com.fossgalaxy.games.fireworks.ai.rule.logic.DeckUtils;
import com.fossgalaxy.games.fireworks.annotations.AgentBuilderStatic;
import com.fossgalaxy.games.fireworks.annotations.AgentConstructor;
import com.fossgalaxy.games.fireworks.state.Card;
import com.fossgalaxy.games.fireworks.state.Deck;
import com.fossgalaxy.games.fireworks.state.GameState;
import com.fossgalaxy.games.fireworks.state.Hand;
import com.fossgalaxy.games.fireworks.state.actions.*;
import com.fossgalaxy.games.fireworks.utils.DebugUtils;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.util.*;

/**
 * Created by WebPigeon on 09/08/2016.
 */
public class MCTS implements Agent {
    public static final int DEFAULT_ITERATIONS = 50_000;
    public static final int DEFAULT_ROLLOUT_DEPTH = 18;
    public static final int DEFAULT_TREE_DEPTH_MUL = 1;
    public static final int NO_LIMIT = 100;
    protected static final boolean OLD_UCT_BEHAVIOUR = false;

    protected final int roundLength;
    protected final int rolloutDepth;
    protected final int treeDepthMul;
    protected final Random random;
    protected final Logger logger = LoggerFactory.getLogger(MCTS.class);

    private final boolean calcTree = false;
    private final boolean generateExamples = false;
    private final String outputFile = "training_data.csv";
    private BufferedWriter fileWriter;

    /**
     * Create a default MCTS implementation.
     * <p>
     * This creates an MCTS agent that has a default roll-out length of 50_000 iterations, a depth of 18 and a tree
     * multiplier of 1.
     */
    public MCTS() {
        this(DEFAULT_ITERATIONS, DEFAULT_ROLLOUT_DEPTH, DEFAULT_TREE_DEPTH_MUL);
    }

    public MCTS(int roundLength) {
        this(roundLength, DEFAULT_ROLLOUT_DEPTH, DEFAULT_TREE_DEPTH_MUL);
    }

    @AgentConstructor("mcts")
    public MCTS(int roundLength, int rolloutDepth, int treeDepthMul) {
        this.roundLength = roundLength;
        this.rolloutDepth = rolloutDepth;
        this.treeDepthMul = treeDepthMul;
        this.random = new Random();
    }

    @AgentBuilderStatic("mctsND")
    public static MCTS buildMCTSND() {
        return new MCTS(MCTS.DEFAULT_ITERATIONS, MCTS.NO_LIMIT, MCTS.NO_LIMIT);
    }

    @Override
    public Action doMove(int agentID, GameState state) {
        long finishTime = System.currentTimeMillis() + 950;
        MCTSNode root = new MCTSNode(
                (agentID + state.getPlayerCount() - 1) % state.getPlayerCount(),
                null,
                Utils.generateAllActions(agentID, state.getPlayerCount())
        );

        // Map each slot in the hand to the list of possible cards that could be in it.
        Map<Integer, List<Card>> possibleCards = DeckUtils.bindCard(agentID, state.getHand(agentID), state.getDeck().toList());

        // Order the slots according to the size of the list of possible cards. Slots with fewer possible cards will
        // appear first in the list.
        List<Integer> bindOrder = DeckUtils.bindOrder(possibleCards);

        if (logger.isTraceEnabled()) {
            logger.trace("Possible bindings: ");
            possibleCards.forEach((slot, cards) -> logger.trace("\t {} {}", slot, DebugUtils.getHistStr(DebugUtils.histogram(cards))));

            // Guaranteed cards
            logger.trace("Guaranteed Cards");

            possibleCards.entrySet().stream()
                    .filter(x -> x.getValue().size() == 1)
                    .forEach(this::printCard);

            logger.trace("We know the value of these");
            possibleCards.entrySet().stream()
                    .filter(x -> x.getValue().stream().allMatch(y -> y.value.equals(x.getValue().get(0).value)))
                    .forEach(this::printCard);

            DebugUtils.printTable(logger, state);
        }

        while(System.currentTimeMillis() < finishTime){
            GameState currentState = state.getCopy();
            IterationObject iterationObject = new IterationObject(agentID);

            // Randomly choose one of the possible cards for each slot and assign it to them.
            // A card that is selected to be in one slot is guaranteed to not be chosen to be in another one.
            Map<Integer, Card> cardsInHand = DeckUtils.bindCards(bindOrder, possibleCards);

            Deck deck = currentState.getDeck();
            Hand myHand = currentState.getHand(agentID);

            // Iterate over all the slots and assign each selected possible card to the slots, but this time using the
            // Hand object. Also, remove each selected possible card from the Deck.
            for (int slot = 0; slot < myHand.getSize(); slot++) {
                Card cardInHand = cardsInHand.get(slot);
                myHand.bindCard(slot, cardInHand);
                deck.remove(cardInHand);
            }
            deck.shuffle();

            MCTSNode current = select(root, currentState, iterationObject);
            int score = rollout(currentState, current);
            current.backup(score);
            if(calcTree){
                System.err.println(root.printD3());
            }
        }

        // Generate Examples
        if (generateExamples) {
            try {
                root.setGameState(state.getCopy());
                File file = new File(outputFile);
                boolean fileExisted = file.exists();
                this.fileWriter = new BufferedWriter(new FileWriter(outputFile, true));
                if (!fileExisted) {
                    writeColumnNames();
                }
                generateExamples(root, agentID);
                this.fileWriter.close();
            } catch (IOException e) {
                System.err.println(e.getMessage());
            }
        }

        if (logger.isInfoEnabled()) {
            for (MCTSNode level1 : root.getChildren()) {
                logger.info("rollout {} moves: max: {}, min: {}, avg: {}, N: {} ", level1.getAction(), level1.rolloutMoves.getMax(), level1.rolloutMoves.getMin(), level1.rolloutMoves.getMean(), level1.rolloutMoves.getN());
                logger.info("rollout {} scores: max: {}, min: {}, avg: {}, N: {} ", level1.getAction(), level1.rolloutScores.getMax(), level1.rolloutScores.getMin(), level1.rolloutScores.getMean(), level1.rolloutScores.getN());
            }
        }

        if (logger.isTraceEnabled()) {
            logger.trace("next player's moves considerations: ");
            for (MCTSNode level1 : root.getChildren()) {
                logger.trace("{}'s children", level1.getAction());
                level1.printChildren();
            }
        }

        Action chosenOne = root.getBestNode().getAction();
        if (logger.isTraceEnabled()) {
            logger.trace("Move Chosen by {} was {}", agentID, chosenOne);
            root.printChildren();
        }
        return chosenOne;
    }

    protected MCTSNode select(MCTSNode root, GameState state, IterationObject iterationObject) {
        MCTSNode current = root;
        int treeDepth = calculateTreeDepthLimit(state);
        boolean expandedNode = false;

        while (!state.isGameOver() && current.getDepth() < treeDepth && !expandedNode) {
            MCTSNode next;
            // If all legal actions from the current node have been generated before, select the node at which we arrive
            // by using UCT for choosing the action we should take.
            if (current.fullyExpanded(state)) {
                next = current.getUCTNode(state);
            }
            // If at least one legal action has not been generated before, expand the current node and set the flag of
            // expanding a node to true.
            else {
                next = expand(current, state);
                expandedNode = true;
            }

            if (next == null) {
                // If all follow on states explored so far are null, we are now a leaf node
                // Ok to early return here - we will have applied current last time round the loop!
                return current;
            }
            // Move one step further in the tree (we move to the node that resulted from the expansion operation or from
            // using the UCT method).
            current = next;

            int agent = current.getAgent();
            int lives = state.getLives();
            int score = state.getScore();

            // This is the action we performed for getting to this state.
            Action action = current.getAction();
            if (action != null) {
                // Apply the action so that the simulated state is affected and the simulated game progresses.
                action.apply(agent, state);
            }

            // Update the values of lives lost and points gain if the action taken was from our agent.
            if (iterationObject.isMyGo(agent)) {
                if (state.getLives() < lives) {
                    iterationObject.incrementLivesLostMyGo();
                }
                if (state.getScore() > score) {
                    iterationObject.incrementPointsGainedMyGo();
                }
            }
        }
        return current;
    }

    protected int calculateTreeDepthLimit(GameState state){
        return (state.getPlayerCount() * treeDepthMul) + 1;
    }

    /**
     * Select a new action for the expansion node.
     *
     * @param state   the game state to travel from
     * @param agentID the AgentID to use for action selection
     * @param node    the Node to use for expansion
     * @return the next action to be added to the tree from this state.
     */
    protected Action selectActionForExpand(GameState state, MCTSNode node, int agentID) {
        Collection<Action> legalActions = node.getLegalMoves(state, agentID);
        if (legalActions.isEmpty()) {
            return null;
        }

        Iterator<Action> actionItr = legalActions.iterator();

        int selected = random.nextInt(legalActions.size());
        Action curr = actionItr.next();
        for (int i = 0; i < selected; i++) {
            curr = actionItr.next();
        }

        return curr;
    }

    protected Action selectActionForRollout(GameState state, int playerID) {
        Collection<Action> legalActions = Utils.generateActions(playerID, state);

        List<Action> listAction = new ArrayList<>(legalActions);
        Collections.shuffle(listAction);

        return listAction.get(0);
    }

    protected MCTSNode expand(MCTSNode parent, GameState state) {
        int nextAgentID = (parent.getAgent() + 1) % state.getPlayerCount();
        // Randomly select one legal action that can be taken from the state, so as to expand the current MCTS node.
        Action action = selectActionForExpand(state, parent, nextAgentID);

        // Return the current node if no legal action was found.
        if (action == null) {
            return parent;
        }
        // If the legal action was already expanded from the current node, return the node which was already a child
        // of the current one.
        if (parent.containsChild(action)) {
            return parent.getChild(action);
        }

        // Creates a child node of the current node that is reached by the legal action obtained.
        GameState stateCopy = state.getCopy();
        action.apply(nextAgentID, stateCopy);
        MCTSNode child = new MCTSNode(
                parent,
                nextAgentID,
                action,
                Utils.generateAllActions(nextAgentID + 1, state.getPlayerCount()),
                stateCopy);

        parent.addChild(child);
        return child;
    }

    protected int rollout(GameState state, MCTSNode current) {
        int playerID = (current.getAgent() + 1) % state.getPlayerCount();
        int moves = 0;

        while (!state.isGameOver() && moves < rolloutDepth) {
            Action action = selectActionForRollout(state, playerID);
            action.apply(playerID, state);
            playerID = (playerID + 1) % state.getPlayerCount();
            moves++;
        }

        current.backupRollout(moves, state.getScore());
        return state.getScore();
    }

    @Override
    public String toString() {
        return "MCTS";
    }

    private void printCard(Map.Entry<Integer, List<Card>> entry) {
        logger.trace("{} : {}", entry.getKey(), entry.getValue());
    }

    public void generateExamples(MCTSNode node, int thisAgentId) throws IOException {
        int totalVisits = node.getChildren().stream().map(c -> c.getVisits()).reduce(0, Integer::sum);

        if (totalVisits == 0)
            return;

        GameState state =  node.getGameState();
        int playerCount = state.getPlayerCount();

        int nextAgentId = (node.getAgent() + 1) % playerCount;
        int nextAgentOffset = 0;
        if (thisAgentId <= nextAgentId)
            nextAgentOffset = nextAgentId - thisAgentId;
        else
            nextAgentOffset = playerCount - thisAgentId + nextAgentId;

        NNState nnState = new NNState(state, nextAgentOffset);

        double[] policy = new double[60];
        Arrays.fill(policy, 0);

        for (MCTSNode child : node.getChildren()) {
            Action actionToGetToChild = child.getAction();
            int actionIdToGetToChild = getActionId(actionToGetToChild, thisAgentId, playerCount);
            policy[actionIdToGetToChild] = (double)child.getVisits() / (double)totalVisits;
        }

        writeRow(nnState, policy);

        for (MCTSNode child : node.getChildren()) {
            if (child.getVisits() != 0)
                generateExamples(child, thisAgentId);
        }
    }

    public int getActionId(Action action, int thisAgentId, int playerCount) {
        if (action instanceof DiscardCard)
            return ((DiscardCard) action).slot;
        if (action instanceof PlayCard)
            return 5 + ((PlayCard) action).slot;

        int actionId = 0;
        int playerToldId = 10;
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

    public int getPlayerOffset(int thisAgentId, int playerId, int playerCount) {
        if (thisAgentId <= playerId)
            return playerId - thisAgentId;
        else
            return playerCount - thisAgentId + playerId;
    }

    public String getCardsColumnNames(String prefix) {
        String cardsNames = "";
        for (int i = 0; i < 5; i++) {
            String cardColor = "";
            switch (i) {
                case 0: { cardColor = "Red"; break; }
                case 1: { cardColor = "Blue"; break; }
                case 2: { cardColor = "Green"; break; }
                case 3: { cardColor = "Orange"; break; }
                case 4: { cardColor = "White"; break; }
            }
            for (int j = 1; j < 6; j++) {
                cardsNames += prefix + cardColor + "_" + Integer.toString(j) + ",";
            }
        }
        return cardsNames;
    }

    public String getActionColumnNames() {
        String actionNames = "";
        for (int i = 0; i < 60; i++) {
            actionNames += "Action_" + Integer.toString(i) + ",";
        }
        return actionNames;
    }

    public void writeColumnNames() throws IOException {

        fileWriter.write(
                "PlayerCount," +
                        "Information," +
                        "CardColorRed," +
                        "CardColorBlue," +
                        "CardColorGreen," +
                        "CardColorOrange," +
                        "CardColorWhite," +
                        "NextAgentOffset," +
                        "Lives," +
                        "Score," +
                        getCardsColumnNames("Deck") +
                        getCardsColumnNames("Discard") +
                        "StateValue," +
                        getActionColumnNames());
        fileWriter.newLine();
    }

    public void writeRow(NNState nnState, double[] policy) throws IOException {
        String row = nnState.toString();
        for (int i = 0; i < policy.length; i++) {
            row += Double.toString(policy[i]) + ",";
        }
        fileWriter.write(row);
        fileWriter.newLine();
    }
}
