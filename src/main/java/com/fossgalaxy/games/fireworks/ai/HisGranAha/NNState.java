package com.fossgalaxy.games.fireworks.ai.HisGranAha;

import com.fossgalaxy.games.fireworks.state.Card;
import com.fossgalaxy.games.fireworks.state.CardColour;
import com.fossgalaxy.games.fireworks.state.Deck;
import com.fossgalaxy.games.fireworks.state.GameState;

import java.util.Collection;

public class NNState {
    private static final int MAX_PLAYER_COUNT = 5;
    private static final int MAX_INFO_TOKENS = 8;
    private static final int MAX_CARD_IN_PILE = 5;
    private static final int MAX_LIVES = 3;
    private static final int MAX_SCORE = 25;
    private static final int MAX_NUM_CARD_1 = 3;
    private static final int MAX_NUM_CARD_5 = 1;
    private static final int MAX_NUM_CARD_REST = 2;
    private static final int NUM_FEATURES = 64;

    int playerCount;
    int information;
    int cardColorRed;
    int cardColorBlue;
    int cardColorGreen;
    int cardColorOrange;
    int cardColorWhite;
    int nextAgentOffset;
    int lives;
    int score;
    int[][] deckCards;
    int[][] discardCards;

    public NNState(GameState gameState, int nextAgentOffset) {
        this.playerCount = gameState.getPlayerCount();
        this.information = gameState.getInfomation();
        this.cardColorRed = gameState.getTableValue(CardColour.RED);
        this.cardColorBlue = gameState.getTableValue(CardColour.BLUE);
        this.cardColorGreen = gameState.getTableValue(CardColour.GREEN);
        this.cardColorOrange = gameState.getTableValue(CardColour.ORANGE);
        this.cardColorWhite = gameState.getTableValue(CardColour.WHITE);
        this.nextAgentOffset = nextAgentOffset;
        this.lives = gameState.getLives();
        this.score = gameState.getScore();
        this.deckCards = new int[5][5];
        setDeckCards(gameState.getDeck());
        this.discardCards = new int[5][5];
        setDiscardCards(gameState.getDiscards());
    }

    public void setDeckCards(Deck deck) {
        setCardsOnMatrix(deck.toList(), this.deckCards);
    }

    public void setDiscardCards(Collection<Card> cards) {
        setCardsOnMatrix(cards, this.discardCards);
    }

    public void setCardsOnMatrix(Collection<Card> cards, int[][] matrix) {
        int i = 0;
        for (Card card : cards) {
            switch (card.colour) {
                case RED: { i = 0; break; }
                case BLUE: { i = 1; break; }
                case GREEN: { i = 2; break; }
                case ORANGE: { i = 3; break; }
                case WHITE: { i = 4; break; }
            }
            matrix[i][card.value - 1]++;
        }
    }

    @Override
    public String toString() {
        return playerCount + "," +
                information + "," +
                cardColorRed + "," +
                cardColorBlue + "," +
                cardColorGreen + "," +
                cardColorOrange + "," +
                cardColorWhite + "," +
                nextAgentOffset + "," +
                lives + "," +
                score + "," +
                matrixToString(deckCards) +
                matrixToString(discardCards);
    }

    public String matrixToString(int[][] matrix) {
        String matrixString = "";
        for (int i = 0; i < matrix.length; i++) {
            for (int j = 0; j < matrix[i].length; j++) {
                matrixString += Integer.toString(matrix[i][j]) + ",";
            }
        }
        return matrixString;
    }

    public double[] getNormalizedFlattenedRepresentation() {
        double[] representation = new double[NUM_FEATURES];
        representation[0] = this.playerCount / ((double) MAX_PLAYER_COUNT);
        representation[1] = this.information / ((double) MAX_INFO_TOKENS);
        representation[2] = this.cardColorRed / ((double) MAX_CARD_IN_PILE);
        representation[3] = this.cardColorBlue / ((double) MAX_CARD_IN_PILE);
        representation[4] = this.cardColorGreen / ((double) MAX_CARD_IN_PILE);
        representation[5] = this.cardColorOrange / ((double) MAX_CARD_IN_PILE);
        representation[6] = this.cardColorWhite / ((double) MAX_CARD_IN_PILE);
        representation[7] = this.lives / ((double) MAX_LIVES);
        representation[8] = this.score / ((double) MAX_SCORE);
        setNormalizedCardCountInRepresentation(9, this.deckCards, representation);
        setNormalizedCardCountInRepresentation(34, this.discardCards, representation);
        representation[59 + nextAgentOffset] = 1;
        return representation;
    }

    public void setNormalizedCardCountInRepresentation(int startIndex, int[][] collection, double[] representation) {
        for (int i = 0; i < collection.length; i++) {
            for (int j = 0; j < collection[i].length; j++) {
                int reprIndex = startIndex + collection[i].length * i + j;
                if (j == 0)
                    representation[reprIndex] = collection[i][j] / ((double) MAX_NUM_CARD_1);
                else if (j == collection[i].length - 1)
                    representation[reprIndex] = collection[i][j] / ((double) MAX_NUM_CARD_5);
                else
                    representation[reprIndex] = ((double) collection[i][j] / MAX_NUM_CARD_REST);
            }
        }
    }
}
