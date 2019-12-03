package com.fossgalaxy.games.fireworks.ai.HisGranAha;

import com.fossgalaxy.games.fireworks.state.GameState;
import com.fossgalaxy.games.fireworks.state.actions.Action;

import java.util.List;
import java.util.Map;


public class NeuralNetwork {
    public static class NeuralNetworkOutput {
        public Map<Action, Double> policy;
        public double value;
    }

    NeuralNetworkOutput predict(GameState state) {
        return new NeuralNetworkOutput();
    }
}
