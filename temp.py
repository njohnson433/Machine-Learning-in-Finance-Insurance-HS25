# ============================================
#   MODEL, OPTIMIZER, LOSS, SCHEDULER
# ============================================


n_epochs = 100
lr = 1e-2
batch_size = 10000
milestones = [100, 500, 1000]

optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=0.1)
model = PoissonNN(input_dim=N, output_dim=M).to(device)
# ============================================
#   TRAINING LOOP (NO DATALOADER)
# ============================================
epoch_losses = []

training_loop = tqdm(range(n_epochs))

for epoch in training_loop:

    perm = torch.randperm(N, device=device)
    total_loss_accum = 0.0
    total_exp_accum = 0.0

    for i in range(0, N, batch_size):
        idx = perm[i:i+batch_size]

        batch_X = X_train_tensor[idx]
        batch_y = y_train_tensor[idx]
        batch_exp = exposure_train_tensor[idx]

        lam = model(batch_X).squeeze()

        loss = loss_fn(lam, batch_y, batch_exp)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        # accumulate weighted losses for reporting
        total_loss_accum += loss.detach()

    epoch_loss = total_loss_accum.item()
    epoch_losses.append(epoch_loss)

    training_loop.set_postfix(loss=epoch_loss)
    scheduler.step()


print("\nTraining completed!\n")


# ============================================
#   EVALUATION (TRAIN + TEST)
# ============================================

model.eval()
with torch.no_grad():
    lam_train = model(X_train_tensor).squeeze().cpu().numpy()
    lam_test = model(X_test_tensor).squeeze().cpu().numpy()

y_train_np = y_train_tensor.cpu().numpy()
y_test_np = y_test_tensor.cpu().numpy()
exp_train_np = exposure_train_tensor.cpu().numpy()
exp_test_np = exposure_test_tensor.cpu().numpy()

eps = 1e-8

# Poisson deviance calculation in numpy
train_dev = 2 * (lam_train - y_train_np + y_train_np * np.log((y_train_np + eps) / (lam_train + eps)))
test_dev = 2 * (lam_test - y_test_np + y_test_np * np.log((y_test_np + eps) / (lam_test + eps)))

#train_poi_dev = np.sum(exp_train_np * train_dev) / np.sum(exp_train_np)
#test_poi_dev = np.sum(exp_test_np * test_dev) / np.sum(exp_test_np)

train_poi_dev = loss_fn(torch.tensor(lam_train), torch.tensor(y_train_np), torch.tensor(exp_train_np)).item() # TODO: This is very strange and needs to be computed with our loss
test_poi_dev = loss_fn(torch.tensor(lam_test), torch.tensor(y_test_np), torch.tensor(exp_test_np)).item()
print("=== TRAIN METRICS ===")
print("MAE:", mean_absolute_error(y_train_np, lam_train, sample_weight=exp_train_np))
print("MSE:", mean_squared_error(y_train_np, lam_train, sample_weight=exp_train_np))
print("Exposure-weighted Poisson deviance (Sklearn):", mean_poisson_deviance(y_train_np, lam_train, sample_weight=exp_train_np))
print("Exposure-weighted Poisson deviance (Torch):", train_poi_dev)

print("\n=== TEST METRICS ===")
print("MAE:", mean_absolute_error(y_test_np, lam_test, sample_weight=exp_test_np))
print("MSE:", mean_squared_error(y_test_np, lam_test, sample_weight=exp_test_np))
print("Exposure-weighted Poisson deviance (Sklearn):", mean_poisson_deviance(y_test_np, lam_test, sample_weight=exp_test_np))
print("Exposure-weighted Poisson deviance (Torch):", test_poi_dev)

# ============================================
#   PLOT TRAINING LOSS
# ============================================

plt.figure(figsize=(10,6))
plt.plot(epoch_losses, label="Exposure-weighted Poisson deviance")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training Loss per Epoch")
plt.grid(True)
plt.legend()
plt.show()


# Free GPU
model = model.to("cpu")
torch.cuda.empty_cache()

