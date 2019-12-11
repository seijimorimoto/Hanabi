package com.fossgalaxy.games.fireworks.ai.HisGranAha;

import com.fossgalaxy.games.fireworks.state.CardColour;
import com.fossgalaxy.games.fireworks.state.GameState;

import java.util.Arrays;

public class NNState {
    public static final int MAX_SCORE = 25;
    private static final int MAX_PLAYER_COUNT = 5;
    private static final int MAX_INFO_TOKENS = 8;
    private static final int MAX_CARDS_IN_PILE = 5;
    private static final int MAX_LIVES = 3;
    private static final int NUM_FEATURES = 18;

    int playerCount;
    int information;
    int nextAgentOffset;
    int lives;
    int[] cardValuesCounts;
    int[] cardColourCounts;
    int score;

    public NNState(GameState gameState, int nextAgentOffset) {
        this.playerCount = gameState.getPlayerCount();
        this.information = gameState.getInfomation();
        this.nextAgentOffset = nextAgentOffset;
        this.lives = gameState.getLives();
        setCardCounts(gameState);
        this.score = gameState.getScore();
    }

    public void setCardCounts(GameState state) {
        CardColour[] colours = { CardColour.RED, CardColour.BLUE, CardColour.GREEN, CardColour.ORANGE, CardColour.WHITE };
        this.cardColourCounts = new int[5];
        this.cardValuesCounts = new int[5];
        for (int i = 0; i < colours.length; i++) {
            int cardValue = state.getTableValue(colours[i]);
            this.cardColourCounts[i] = cardValue;
            for (int j = 0; j < cardValue; j++) {
                this.cardValuesCounts[j]++;
            }
        }
    }

    public double[] getNormalizedFlattenedRepresentation() {
        double[] representation = new double[NUM_FEATURES];
        representation[0] = this.playerCount / ((double) MAX_PLAYER_COUNT);
        representation[1] = this.information / ((double) MAX_INFO_TOKENS);
        representation[2] = this.lives / ((double) MAX_LIVES);
        setNormalizedCardCountInRepresentation(3, cardValuesCounts, representation);
        setNormalizedCardCountInRepresentation(8, cardColourCounts, representation);
        representation[13 + nextAgentOffset] = 1;
        return representation;
    }

    public void setNormalizedCardCountInRepresentation(int startIndex, int[] collection, double[] representation) {
        for (int i = 0; i < collection.length; i++) {
            representation[startIndex + i] = collection[i] / ((double) MAX_CARDS_IN_PILE);
        }
    }

    @Override
    public String toString() {
        return playerCount + "," +
                information + "," +
                nextAgentOffset + "," +
                lives + "," +
                String.join(",", Arrays.stream(cardValuesCounts).mapToObj(String::valueOf).toArray(String[]::new)) + "," +
                String.join(",", Arrays.stream(cardColourCounts).mapToObj(String::valueOf).toArray(String[]::new)) + "," +
                score + ",";
    }

    @Override
    public boolean equals(Object obj) {
        if (this == obj)
            return true;
        if (obj == null)
            return false;
        if (getClass() != obj.getClass())
            return false;
        NNState other = (NNState) obj;
        if (playerCount != other.playerCount)
            return false;
        if (information != other.information)
            return false;
        if (nextAgentOffset != other.nextAgentOffset)
            return false;
        if (lives != other.lives)
            return false;
        if (!Arrays.equals(cardValuesCounts, other.cardValuesCounts))
            return false;
        if (!Arrays.equals(cardColourCounts, other.cardColourCounts))
            return false;
        if (score != other.score)
            return false;
        return true;
    }

    @Override
    public int hashCode() {
        final int prime = 31;
        int result = 1;
        result = prime * result + Integer.hashCode(playerCount);
        result = prime * result + Integer.hashCode(information);
        result = prime * result + Integer.hashCode(nextAgentOffset);
        result = prime * result + Integer.hashCode(lives);
        result = prime * result + Arrays.hashCode(cardValuesCounts);
        result = prime * result + Arrays.hashCode(cardColourCounts);
        result = prime * result + Integer.hashCode(score);
        return result;
    }
}
