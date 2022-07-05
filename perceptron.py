# from Typing import List


class Perceptron:
    def __init__(self, weights, threshold):
        """

        Args:
            weights (List): Perceptron weights
            threshold (float): Decision threshold for output
        """
        self.currInput = []
        self.threshold = threshold
        self.weights = weights

    def forward(self, data):
        if len(data) != len(self.weights):
            raise ValueError("Weights and data should be the same size")

        output = 0
        for i, (x, y) in enumerate(zip(self.weights, data)):
            output += x * y

        self.currInput = data

        return 1 if output >= self.threshold else 0


class Optimizer:
    def __init__(self, lr):
        """Optimizer for training the perceptron.

        Implements the formula: weight(t+1) = weight(t) + (lr * ((target - actualOutput) * input))

        Args:
            lr (float): Learning rate
        """

        self.lr = lr

    def step(self, currInput, weights, target, output):
        """Takes an optimizer step

        Args:
            currInput (List): List of inputs e.g (0, 1)
            weights (List): List of weights e.g [0.2, 0.5]
            target (int): Actual target
            output (int): Actual utput

        Returns:
            List: List of updated weights
        """

        newWeights = []
        for input_, weight in zip(currInput, weights):

            newWeight = self.computeNewWeight(weight, input_, target, output)
            newWeights.append(newWeight)

        return newWeights

    def computeNewWeight(self, oldWeight, sampleInput, target, output):

        newWeight = oldWeight + self.lr * ((target - output) * sampleInput)

        return round(newWeight, 2)


class Trainer:
    def __init__(self, initialWeights, threshold, learningRate):
        """_summary_

        Args:
            initialWeights (List): previous weight to be updated
            threhold (int): decision threhold for classification of output into 0 or 1
            learningRate (float): Learining rate
        """

        self.model = Perceptron(initialWeights, threshold)
        self.optimizer = Optimizer(learningRate)

    def train(self, batch, target, maxEpoch=10):
        """Train the perceptron model

        Args:
            batch (List[List]): The truth table input
            target (List[int]): target values
            maxEpoch (int, optional): Maximujm training epochs. Defaults to 10.
        """

        allCorrect = False
        epoch = 1
        while not allCorrect and epoch < maxEpoch:

            print("==========================================================")
            print("==========================================================")
            print("..........................................................")

            print(f"Epoch {epoch}")

            correct = 0
            for i, data in enumerate(batch):
                actualOutput = self.model.forward(data)

                if actualOutput == target[i]:
                    correct += 1

                newWeights = self.optimizer.step(
                    self.model.currInput,
                    self.model.weights,
                    target[i],
                    actualOutput,
                )

                print(
                    f"Actual output: {actualOutput}, target output: {target[i]}, Old weights: {self.model.weights}, Updated weights: {newWeights}"
                )

                self.model.weights = newWeights

            if correct == len(batch):
                allCorrect = True
            epoch += 1


if __name__ == "__main__":

    print("**********************************************************")
    print("***********************Problem 2.1 ***********************")
    print("**********************************************************")

    inputProblem21 = [(0, 0), (0, 1), (1, 0), (1, 1)]
    targetProblem21 = [0, 1, 1, 1]
    weightsProblem21 = [0.5, 0.8]
    threholdProblem21 = 2.9
    lrProblem21 = 0.4

    trainer = Trainer(weightsProblem21, threholdProblem21, lrProblem21)
    trainer.train(inputProblem21, targetProblem21)

    print("**********************************************************")
    print("***********************Problem 2.2 ***********************")
    print("**********************************************************")

    inputProblem22 = [
        (0, 0, 0),
        (1, 0, 0),
        (0, 1, 0),
        (1, 1, 0),
        (0, 0, 1),
        (1, 0, 1),
        (0, 1, 1),
        (1, 1, 1),
    ]
    targetProblem22 = [0, 1, 0, 1, 0, 1, 0, 1]
    weightsProblem22 = [0.6, 0.8, 0.9]
    threholdProblem22 = -1.0
    lrProblem22 = 0.4

    trainer = Trainer(weightsProblem22, threholdProblem22, lrProblem22)
    trainer.train(inputProblem22, targetProblem22)
