package exemplo;

import org.encog.engine.network.activation.ActivationLinear;
import org.encog.engine.network.activation.ActivationSigmoid;
import org.encog.ml.data.MLData;
import org.encog.ml.data.MLDataPair;
import org.encog.ml.data.MLDataSet;
import org.encog.ml.data.basic.BasicMLDataSet;
import org.encog.ml.train.MLTrain;
import org.encog.neural.networks.BasicNetwork;
import org.encog.neural.networks.layers.BasicLayer;
import org.encog.neural.networks.training.propagation.resilient.ResilientPropagation;

public class RNA_DE_NOOB {

	public static void main(String[] args) {
		double[][] entrada = { { 0, 1 ,2}, { 1,  2,3}, { 1, 3,4 }, { 0, 4,5} };
		double[][] saida = { { 5 }, { 8 }, { 11 }, { 14 } };
		
		BasicNetwork network = new BasicNetwork();
		network.addLayer(new BasicLayer(entrada[0].length));
		//network.addLayer(new BasicLayer(new ActivationSigmoid(), true, 3));
		network.addLayer(new BasicLayer(new ActivationLinear(), false, 1));
		network.getStructure().finalizeStructure();
		network.reset();

		MLDataSet trainingSet = new BasicMLDataSet(entrada, saida);
		MLTrain train = new ResilientPropagation(network, trainingSet);

		int epoch = 1;
		do {
			train.iteration();
			System.out.println("Epoch #" + epoch + " Erro:" + train.getError());
			epoch++;
		} while (epoch < 2000);

		System.out.println("Neural Network Results : ");
		for (MLDataPair pair : trainingSet) {
			MLData output = network.compute(pair.getInput());
			System.out.println(pair.getInput().getData(0) + " , " + pair.getInput().getData(1) + " , actual="
					+ output.getData(0) + " , ideal=" + pair.getIdeal().getData(0));
			
		}

	}

}
