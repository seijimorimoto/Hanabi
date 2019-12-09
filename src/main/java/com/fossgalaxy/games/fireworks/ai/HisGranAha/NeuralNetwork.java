package com.fossgalaxy.games.fireworks.ai.HisGranAha;

import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.records.reader.impl.csv.CSVRecordReader;
import org.datavec.api.split.FileSplit;
import org.deeplearning4j.datasets.datavec.RecordReaderMultiDataSetIterator;
import org.deeplearning4j.nn.conf.ComputationGraphConfiguration;
import org.deeplearning4j.nn.conf.InputPreProcessor;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.*;
import org.deeplearning4j.nn.conf.preprocessor.CnnToFeedForwardPreProcessor;
import org.deeplearning4j.nn.conf.preprocessor.FeedForwardToCnnPreProcessor;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.modelimport.keras.KerasModelImport;
import org.deeplearning4j.nn.modelimport.keras.exceptions.InvalidKerasConfigurationException;
import org.deeplearning4j.nn.modelimport.keras.exceptions.UnsupportedKerasConfigurationException;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.iterator.MultiDataSetIterator;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import java.io.File;
import java.io.IOException;
import java.util.HashMap;
import java.util.Map;


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
        inputsAsINDArray = Nd4j.expandDims(inputsAsINDArray, 0);
        INDArray[] outputs = this.model.output(inputsAsINDArray);
        double[] policy = outputs[0].toDoubleVector();
        double value = outputs[1].toDoubleVector()[0];
        return new NeuralNetworkOutput(policy, value);
    }

    public void compile() {
        Map<String, InputPreProcessor> preProcessorMap = new HashMap<>();
        preProcessorMap.put("Conv1D_1", new FeedForwardToCnnPreProcessor(64, 1, 1));
        preProcessorMap.put("Dense1", new CnnToFeedForwardPreProcessor(60, 512, 1));

        ComputationGraphConfiguration.GraphBuilder confBuilder = new NeuralNetConfiguration.Builder()
                .updater(new Adam(0.001))
                .graphBuilder()
                .addInputs("input")
                .addLayer("Conv1D_1", new Convolution1D.Builder().kernelSize(64).stride(3).nIn(1).nOut(64).hasBias(false).activation(Activation.RELU).build(), "input")
                .addLayer("BatchNorm1", new BatchNormalization.Builder().nIn(64).nOut(64).build(), "Conv1D_1")
                .addLayer("Conv1D_2", new Convolution1D.Builder().kernelSize(128).stride(3).nIn(64).nOut(64).hasBias(false).activation(Activation.RELU).build(), "BatchNorm1")
                .addLayer("BatchNorm2", new BatchNormalization.Builder().nIn(64).nOut(64).build(), "Conv1D_2")
                .addLayer("Conv1D_3", new Convolution1D.Builder().kernelSize(256).stride(3).nIn(64).nOut(64).hasBias(false).activation(Activation.RELU).build(), "BatchNorm2")
                .addLayer("BatchNorm3", new BatchNormalization.Builder().nIn(64).nOut(64).build(), "Conv1D_3")
                .addLayer("Conv1D_4", new Convolution1D.Builder().kernelSize(512).stride(3).nIn(64).nOut(64).hasBias(false).activation(Activation.RELU).build(), "BatchNorm3")
                .addLayer("BatchNorm4", new BatchNormalization.Builder().nIn(64).nOut(64).build(), "Conv1D_4")
                .addLayer("Dense1", new DenseLayer.Builder().nIn(32768).nOut(1024).hasBias(false).activation(Activation.RELU).build(), "BatchNorm4")
                .addLayer("BatchNormD1", new BatchNormalization.Builder().nIn(1024).nOut(1024).build(), "Dense1")
                .addLayer("Drop1", new DropoutLayer.Builder().dropOut(0.7).nIn(1024).nOut(1024).build(), "BatchNormD1")
                .addLayer("Dense2", new DenseLayer.Builder().nIn(1024).nOut(512).hasBias(false).activation(Activation.RELU).build(), "Drop1")
                .addLayer("BatchNormD2", new BatchNormalization.Builder().nIn(512).nOut(512).build(), "Dense2")
                .addLayer("Drop2", new DropoutLayer.Builder().dropOut(0.7).nIn(512).nOut(512).build(), "BatchNormD2")
                .addLayer("Pi", new OutputLayer.Builder().nIn(512).nOut(60).activation(Activation.SOFTMAX).lossFunction(LossFunctions.LossFunction.MCXENT).build(), "Drop2")
                .addLayer("V", new OutputLayer.Builder().nIn(512).nOut(1).activation(Activation.RELU).lossFunction(LossFunctions.LossFunction.MSE).build(), "Drop2")
                .setOutputs("Pi", "V");
        confBuilder.setInputPreProcessors(preProcessorMap);

        this.model = new ComputationGraph(confBuilder.build());
    }

    public void importKerasModel(String modelConfigPath, String modelWeightsPath)
            throws IOException, InvalidKerasConfigurationException, UnsupportedKerasConfigurationException {
        this.model = KerasModelImport.importKerasModelAndWeights(modelConfigPath, modelWeightsPath);
    }

    public void setTrainData(String trainFeaturesPath, String trainLabelsPath, int numLinesSkip, int batchSize)
            throws InterruptedException, IOException {
        String delimiter = ",";
        RecordReader trainFeaturesReader = new CSVRecordReader(numLinesSkip, delimiter);
        trainFeaturesReader.initialize(new FileSplit(new File(trainFeaturesPath)));
        RecordReader trainLabelsReader = new CSVRecordReader(numLinesSkip, delimiter);
        trainLabelsReader.initialize(new FileSplit(new File(trainLabelsPath)));

        this.modelIterator = new RecordReaderMultiDataSetIterator.Builder(batchSize)
                .addReader("trainFeatures", trainFeaturesReader)
                .addReader("trainLabels", trainLabelsReader)
                .addInput("trainFeatures")
                .addOutput("trainLabels", 0, 59)
                .addOutput("trainLabels", 60, 60)
                .build();
    }

    public void train(int numEpochs) {
        this.model.fit(modelIterator, numEpochs);
    }
}
