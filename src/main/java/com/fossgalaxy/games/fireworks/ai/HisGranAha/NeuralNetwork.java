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

/**
 * Container of a neural network model designed for Hanabi and represented internally as a DL4J
 * ComputationGraph.
 */
public class NeuralNetwork {
    /**
     * Container for the outputs of the neural network.
     */
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

    /**
     * Makes a prediction based on an appropriate representation of a state of the game.
     * @param state The state of the game from which to make a prediction.
     * @return The output of the prediction.
     */
    public NeuralNetworkOutput predict(NNState state) {
        double[] inputs = state.getNormalizedFlattenedRepresentation();
        return predict(inputs);
    }

    /**
     * Makes a prediction based on a vector containing double precision values for each of the
     * features of the game state considered as inputs to the internal neural network model.
     * @param stateFeatures The vector of values for each game state feature.
     * @return The output of the prediction.
     */
    public NeuralNetworkOutput predict(double[] stateFeatures) {
        INDArray inputsAsINDArray = Nd4j.create(stateFeatures, new int[]{stateFeatures.length});
        // Increment the dimensionality by 1 to represent the batch_size, as that is how DL4J
        // expects the inputs.
        inputsAsINDArray = Nd4j.expandDims(inputsAsINDArray, 0);
        INDArray[] outputs = this.model.output(inputsAsINDArray);
        double[] policy = outputs[0].toDoubleVector();
        double value = outputs[1].toDoubleVector()[0] * NNState.MAX_SCORE;
        return new NeuralNetworkOutput(policy, value);
    }

    /**
     * Creates the internal neural network model using DL4J API calls.
     */
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

    /**
     * Imports a neural network model that was built using the Keras API in Python.
     * @param modelConfigPath The path to the configuration of the model.
     * @param modelWeightsPath The path to the saved weights and parameters of the model.
     * @throws IOException One of the paths was not found or could not be opened.
     * @throws InvalidKerasConfigurationException The imported Keras model is invalid.
     * @throws UnsupportedKerasConfigurationException The imported Keras model is not supported by DL4J.
     */
    public void importKerasModel(String modelConfigPath, String modelWeightsPath)
            throws IOException, InvalidKerasConfigurationException, UnsupportedKerasConfigurationException {
        this.model = KerasModelImport.importKerasModelAndWeights(modelConfigPath, modelWeightsPath);
    }

    /**
     * Imports a neural network model that was built using the DL4J API.
     * @param modelPath The path to the saved model.
     * @param loadForRetraining Whether to load the model for retraining it or just for inference.
     * @throws IOException The path to the saved model was not found or could not be opened.
     */
    public void importDL4JModel(String modelPath, boolean loadForRetraining) throws IOException
    {
        this.model = ComputationGraph.load(new File(modelPath), loadForRetraining);
    }

    /**
     * Exports the internal neural network model to secondary memory.
     * @param savePath The path to the location where the model is to be saved.
     * @param canBeRetrained Whether to allow the model to be retrained if imported in the future.
     * @throws IOException The path to the save path was not found or could not be accessed.
     */
    public void saveModel(String savePath, boolean canBeRetrained) throws IOException
    {
        this.model.save(new File(savePath), canBeRetrained);
    }

    /**
     * Sets the data that will be used for training the internal neural network.
     * @param trainDataSetPath The path to the location of the training data file.
     * @param numLinesSkip Number of lines in the training data file that must be skipped before
     *                     reaching the actual data.
     * @param batchSize The number of training examples that will be used before performing an
     *                  update to the trainable parameters.
     * @throws InterruptedException An operation was inadvertently interrupted.
     * @throws IOException The path to the training data file was not found or could not be opened.
     */
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

    /**
     * Trains the internal neural network model.
     * @param numEpochs The number of epochs to train the model.
     */
    public void train(int numEpochs) {
        this.model.fit(modelIterator, numEpochs);
    }
}
