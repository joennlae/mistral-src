import torch
import os
import math

path = 'scores_2'

scores = []
scores_softmax = []
actual_scores = []
for i in range(1, 10):
    scores_tensor = torch.load(f'{path}/scores_{i}.pt')
    scores.append(scores_tensor)
    scores_softmax_tensor = torch.load(f'{path}/scoressoftmax_{i}.pt')
    scores_softmax.append(scores_softmax_tensor)
    actual_scores_tensor = torch.load(f'{path}/actualscores_{i}.pt')
    actual_scores.append(actual_scores_tensor)

print(scores[0].shape)
print(scores_softmax[0].shape)
print(actual_scores[0].shape)

print(actual_scores[0][0][0])
print(scores[0][0][0])
print(scores_softmax[0][0][0])

test_scores = actual_scores[0]
print(test_scores[0][0])

values_per_row = torch.Tensor([1, 2, 3, 4, 5])
values_per_row = 1 / values_per_row
values_per_row = values_per_row.view(1, 5).T
print(values_per_row)

scores_softmax_test = torch.nn.functional.softmax(test_scores, dim=-1)
print(scores_softmax_test[0][0])

# apply function for each element in tensor
def wild_activation(x):
    if x < -4:
        return 0
    elif x < 0:
        return 0.001
    else:
        return math.pow(x, 2)

test_scores_cpu = test_scores.cpu()
wild_out = test_scores_cpu.apply_(wild_activation)
wild_out = wild_out.to("cuda")

print(wild_out[0][0])
scores_test = wild_out #  * values_per_row.to("cuda")
sums = scores_test.sum(dim=-1)
sums = sums[..., None]
print(sums.shape, scores_test.shape)
scores_test = (scores_test) / sums
print(scores_test[0][0])