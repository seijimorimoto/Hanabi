package com.fossgalaxy.games.fireworks.ai.HisGranAha;

import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.records.reader.impl.csv.CSVRecordReader;
import org.datavec.api.split.FileSplit;
import org.deeplearning4j.datasets.datavec.RecordReaderMultiDataSetIterator;
import org.deeplearning4j.nn.conf.ComputationGraphConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.modelimport.keras.KerasModelImport;
import org.deeplearning4j.nn.modelimport.keras.exceptions.InvalidKerasConfigurationException;
import org.deeplearning4j.nn.modelimport.keras.exceptions.UnsupportedKerasConfigurationException;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.iterator.MultiDataSetIterator;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import java.io.File;
import java.io.IOException;


public class NeuralNetwork {
    public static class NeuralNetworkOutput {
        public double[] policy;
        public double value;

        public NeuralNetworkOutput(double[] policy, double value) {
            this.policy = policy;
            this.value = value;
        }
    }

    private ComputationGraph model;
    private MultiDataSetIterator modelIterator;

    public NeuralNetworkOutput predict(NNState state) {
        double[] inputs = state.getNormalizedFlattenedRepresentation();
        return predict(inputs);
    }

    public NeuralNetworkOutput predict(double[] stateFeatures) {
        INDArray inputsAsINDArray = Nd4j.create(stateFeatures, new int[]{stateFeatures.length});
        inputsAsINDArray = Nd4j.expandDims(inputsAsINDArray, 0);
        INDArray[] outputs = this.model.output(inputsAsINDArray);
        double[] policy = outputs[0].toDoubleVector();
        double value = outputs[1].toDoubleVector()[0] * NNState.MAX_SCORE;
        return new NeuralNetworkOutput(policy, value);
    }

    public void compile() {
        ComputationGraphConfiguration.GraphBuilder confBuilder = new NeuralNetConfiguration.Builder()
                .updater(new Adam(0.001))
                .graphBuilder()
                .addInputs("input")
                .addLayer("Dense1", new DenseLayer.Builder().nIn(18).nOut(32).activation(Activation.RELU).build(), "input")
                .addLayer("Dense2", new DenseLayer.Builder().nIn(32).nOut(16).activation(Activation.RELU).build(), "Dense1")
                .addLayer("Pi", new OutputLayer.Builder().nIn(16).nOut(60).activation(Activation.SOFTMAX).lossFunction(LossFunctions.LossFunction.MCXENT).build(), "Dense2")
                .addLayer("V", new OutputLayer.Builder().nIn(16).nOut(1).activation(Activation.SIGMOID).lossFunction(LossFunctions.LossFunction.MSE).build(), "Dense2")
                .setOutputs("Pi", "V");
        this.model = new ComputationGraph(confBuilder.build());
        this.model.init();
        this.model.setListeners(new ScoreIterationListener(1000));
    }

    public void importKerasModel(String modelConfigPath, String modelWeightsPath)
            throws IOException, InvalidKerasConfigurationException, UnsupportedKerasConfigurationException {
        this.model = KerasModelImport.importKerasModelAndWeights(modelConfigPath, modelWeightsPath);
    }

    public void importDL4JModel(String modelPath, boolean loadForRetraining) throws IOException
    {
        this.model = ComputationGraph.load(new File(modelPath), loadForRetraining);
    }

    public void saveModel(String savePath, boolean canBeRetrained) throws IOException
    {
        this.model.save(new File(savePath), canBeRetrained);
    }

    public void setTrainData(String trainDataSetPath, int numLinesSkip, int batchSize)
            throws InterruptedException, IOException {
        String delimiter = ",";
        RecordReader trainReader = new CSVRecordReader(numLinesSkip, delimiter);
        trainReader.initialize(new FileSplit(new File(trainDataSetPath)));

        this.modelIterator = new RecordReaderMultiDataSetIterator.Builder(batchSize)
                .addReader("trainReader", trainReader)
                .addInput("trainReader", 0, 17)
                .addOutput("trainReader", 19, 78)
                .addOutput("trainReader", 18, 18)
                .build();
    }

    public void train(int numEpochs) {
        this.model.fit(modelIterator, numEpochs);
    }
}
