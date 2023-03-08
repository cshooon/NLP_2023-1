# NLP_2023-1
자연어처리 수업 과제입니다.
## Lab 1
기존 코드에서 dataset을 Stanford Sentiment Treebank로 변경했습니다.
* feedback -> label
* verified_reviews -> sentence
## Lab 2
기존 코드에서 # 부분 코드 입력
```python
# Step 0: Get the data
x_data, y_target = get_toy_data(batch_size)

# Step 1: Clear the gradients
perceptron.zero_grad()

# Step 2: Compute the forward pass of the model
y_pred = perceptron(x_data).squeeze()

# Step 3: Compute the loss value that we wish to optimize
loss = bce_loss(y_pred, y_target)

# Step 4: Propagate the loss signal backward
loss.backward()

# Step 5: Trigger the optimizer to perform one update
optimizer.step()
```
