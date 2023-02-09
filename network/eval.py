import numpy as np
from statistics import mean

def calc_accuracy(nn, test_batch, test_labels, print_acc=False):
    loss = []

    correct = 0
    nsamples = len(test_batch)
    for sample, label in zip(nn.predict(test_batch), test_labels): 
        prediction = np.abs(np.round(sample))[0]
        loss.append(
            nn.loss.function(label, sample)
        )
        if np.array_equal(label, prediction):
            correct += 1 

    accuracy = correct / nsamples * 100
    if print_acc: 
        print(f"\nAcuracy: {accuracy:.2f}% ({correct}/{nsamples})")
        print(f"Test Loss: {mean(loss)}")
    return accuracy
