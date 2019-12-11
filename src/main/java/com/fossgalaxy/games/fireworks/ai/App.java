package com.fossgalaxy.games.fireworks.ai;

import com.fossgalaxy.games.fireworks.GameRunner;
import com.fossgalaxy.games.fireworks.GameStats;
import com.fossgalaxy.games.fireworks.players.Player;
import com.fossgalaxy.games.fireworks.utils.AgentUtils;
import com.fossgalaxy.stats.BasicStats;
import com.fossgalaxy.stats.StatsSummary;

import java.util.Random;

/**
 * Game runner for testing.
 *
 * This will run a bunch of games with your agent so you can see how it does.
 */
public class App 
{
    public static void main( String[] args )
    {
        int numPlayers = 5;
        int numGames = 1;
        String[] agentOthers = { "iggi", "piers", "flawed", "outer", "vdb-paper", "legal_random"};
        String agentOurs = "HisGranAha";

        Random random = new Random();
        StatsSummary statsSummary = new BasicStats();

        for (int i = 0; i < numGames; i++) {
            for (int j = 0; j < agentOthers.length; j++) {
                GameRunner runner = new GameRunner("test-game", numPlayers);
                int positionOurs = random.nextInt(numPlayers);
                for (int k = 0; k < numPlayers; k++) {
                    if (k == positionOurs) {
                        Player player = new AgentPlayer(agentOurs, AgentUtils.buildAgent(agentOurs));
                        runner.addNamedPlayer(agentOurs, player);
                    } else {
                        Player player = new AgentPlayer(agentOthers[j], AgentUtils.buildAgent(agentOthers[j]));
                        runner.addNamedPlayer(agentOthers[j], player);
                    }
                }
                GameStats stats = runner.playGame(random.nextLong());
                statsSummary.add(stats.score);
            }
        }

        //print out the stats
        System.out.println(String.format("Our agent: Avg: %f, min: %f, max: %f",
                statsSummary.getMean(),
                statsSummary.getMin(),
                statsSummary.getMax()));
    }
}
