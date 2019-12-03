package com.fossgalaxy.games.fireworks.ai.HisGranAha;

import com.fossgalaxy.games.fireworks.ai.Agent;
import com.fossgalaxy.games.fireworks.ai.iggi.Utils;
import com.fossgalaxy.games.fireworks.ai.mcts.IterationObject;
import com.fossgalaxy.games.fireworks.ai.rule.logic.DeckUtils;
import com.fossgalaxy.games.fireworks.ai.sample.SampleAgents;
import com.fossgalaxy.games.fireworks.annotations.AgentBuilderStatic;
import com.fossgalaxy.games.fireworks.state.actions.Action;
import com.fossgalaxy.games.fireworks.state.Card;
import com.fossgalaxy.games.fireworks.state.Deck;
import com.fossgalaxy.games.fireworks.state.GameState;
import com.fossgalaxy.games.fireworks.state.Hand;
import com.fossgalaxy.games.fireworks.utils.DebugUtils;

import java.util.*;

public class HisGranAha implements Agent {
    public static final int DEFAULT_ITERATIONS = 50_000;
    public static final int DEFAULT_ROLLOUT_DEPTH = 18;
    public static final int DEFAULT_TREE_DEPTH_MUL = 1;
    public static final int NO_LIMIT = 100;
    public static final int TIME_LIMIT = 1000;
    protected static final boolean OLD_UCT_BEHAVIOUR = false;

    protected final int roundLength;
    protected final int rolloutDepth;
    protected final int treeDepthMul;
    private Random random;
    private NeuralNetwork nn;

    public HisGranAha(int roundLength, int rolloutDepth, int treeDepthMul) {
        this.roundLength = roundLength;
        this.rolloutDepth = rolloutDepth;
        this.treeDepthMul = treeDepthMul;
        this.random = new Random();
        nn = new NeuralNetwork();
    }

    @Override
    public Action doMove(int agentID, GameState state) {
        long finishTime = System.currentTimeMillis() + TIME_LIMIT;
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

        while (System.currentTimeMillis() < finishTime) {
            GameState currentState = state.getCopy();
            IterationObject iterationObject = new IterationObject(agentID);

            // Randomly choose one of the possible cards for each slot and assign it to them.
            // A card that is selected to be in one slot is guaranteed to not be chosen to be in another one.
            Map<Integer, Card> cardsInHand = DeckUtils.bindCards(bindOrder, possibleCards);

            Deck deck = currentState.getDeck();
            Hand myHand = currentState.getHand(agentID);
 
            // Iterate over all the slots and assign each selected possible card to the slots, but this time using the
            // Hand object. Also, remove each selected possible card from the Deck.
            for (int slot = 0; slot < myHand.getSize(); slot++)
            {
                Card cardInHand = cardsInHand.get(slot);
                myHand.bindCard(slot, cardInHand);
                deck.remove(cardInHand);
            }
            deck.shuffle();

            MCTSNode current = select(root, currentState, iterationObject);

        }
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
        MCTSNode child = new MCTSNode(parent, nextAgentID, action, Utils.generateAllActions(nextAgentID, state.getPlayerCount()));

        // Neural network prediction.
        NeuralNetwork.NeuralNetworkOutput outputs = nn.predict(state);
        child.setScore(outputs.value);
        // Update Q-Values

        parent.addChild(child);
        return child;
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
}
