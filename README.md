# 🧠 Neura — A Minimal Deep Learning Framework (WIP)

**Neura** is a lightweight, from-scratch deep learning framework written in Python. Built to be simple, readable, and extensible, it's designed for educational clarity and hardcore research flexibility.

Currently supports:
- ✅ Forward & backward tensor ops
- ✅ Autograd engine
- ✅ Optimizers (SGD, Adam)
- ✅ Basic layers (Linear, ReLU, etc.)
- ✅ Full training loop for MNIST (CPU only)

> ⚙️ **GPU backend coming soon — CUDA integration starts now.**

---

## 🚀 Why Neura?

Most deep learning frameworks are bloated, abstracted, and optimized for scale — not understanding. Neura goes the other way:

- **Zero dependencies** (no PyTorch, no TensorFlow)
- **Every component hand-coded**, from autograd to model loop
- Easy to hack, extend, or rewrite
- Built for researchers, tinkerers, and learners who want control

---

## 🧪 Current Capabilities

| Feature                  | Status            |
|--------------------------|-------------------|
| Tensor operations        | ✅ Implemented     |
| Autograd engine          | ✅ Implemented     |
| CPU backend              | ✅ Working         |
| Basic layers (Linear, etc.) | ✅ Done         |
| Optimizers (SGD, Adam)   | ✅ Working           |
| Loss functions           | ✅ Working            |
| MNIST training loop      | ✅ Working         |
| GPU backend (CUDA)       | 🚧 In progress     |
| Convolutional layers     | ✅ Done         |
| Model saving/loading     | 🕒 Planned         |

---

## 📦 Example: Train a model on MNIST

```python
import neura.nn as nn
import neura.optim as optim
from neura.data import Dataset, DataLoader

class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = Linear(784, 128)
        self.relu = ReLU()
        self.fc2 = Linear(128, 10)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

# Instantiate model
model = MLP()

# Load data
train_loader, test_loader = DataLoader(dataset, batch_size=32, shuffle=True)

# Define loss and optimizer
loss_fn = BCEWithLogitLoss()
optimizer = Adam(model.parameters(), lr=0.01)

for epoch in range(5):
    for x_batch, y_batch in train_loader:
        preds = model(x_batch)
        loss = loss_fn(preds, y_batch)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch+1} complete")
```

## 🔮 Roadmap

- [ ] **GPU backend** with CUDA / CuPy
- [ ] **Convolutional layers** (Conv2D, MaxPool, etc.)
- [ ] **Transformer components** (Multihead Attention, LayerNorm)
- [ ] **Checkpointing and model loading**
- [ ] **Training benchmark suite** (compare against PyTorch, NumPy baselines)
- [ ] **Config system / CLI launcher**
- [ ] **NeuraScript** — DSL for fast, modular model definition

---


## 👨‍💻 Author

**Yusuf Mohamed**  
AI Researcher | ML Engineer | Open-source Contributor

- Currently building **Neura**, a full deep learning stack from scratch  
- Creator of **GRF**, a new multimodal fusion model  
- Contributor at **Hugging Face**  
- Passionate about building foundational tools for open, transparent AI

📎 [GitHub – @yushi2006](https://github.com/yushi2006)  
📎 [LinkedIn – Yusuf Mohamed](https://www.linkedin.com/in/yusufmohamed2006/)

---

## 🧠 Vision

Neura is more than a framework — it’s a tool for researchers who want to go *deeper*. It’s about:

- Understanding DL by building it
- Controlling every layer of abstraction
- Creating custom architectures without 20 wrappers and 500MB of dependencies
- Creating Custom CUDA kernels with python


If PyTorch is a spaceship, Neura is the engine on the table — raw, flexible, and yours to optimize.

> Clone. Hack. Break. Rebuild. That’s the point.

---

## 🤝 Contributing

Neura is a one-man project (for now), but it’s open to contributions. Whether you:
- Want to build CUDA kernels
- Improve layer coverage
- Add test coverage or infra
- Or just use it and report bugs

…you’re welcome here.

**Open an issue. Fork the repo. Let's build.**

---

## 📄 License

**MIT License** — free to use, modify, and commercialize with attribution.
