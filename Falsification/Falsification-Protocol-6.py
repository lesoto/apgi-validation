from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score


class APGIInspiredNetwork(nn.Module):
    """
    Neural network with APGI architectural constraints

    Key features:
    1. Separate exteroceptive and interoceptive pathways
    2. Learned precision weighting
    3. Threshold-gated global workspace
    4. Somatic marker integration
    """

    def __init__(self, config: Dict):
        super().__init__()
        self.config = config  # Store config as instance variable

        # =====================
        # EXTEROCEPTIVE PATHWAY
        # =====================
        self.extero_encoder = nn.Sequential(
            nn.Linear(config["extero_dim"], 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
        )

        # =====================
        # INTEROCEPTIVE PATHWAY
        # =====================
        self.intero_encoder = nn.Sequential(
            nn.Linear(config["intero_dim"], 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
        )

        # =====================
        # PRECISION NETWORKS
        # =====================
        # Learn to estimate precision from context
        self.Pi_e_network = nn.Sequential(
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Softplus(),  # Ensure positive
        )

        self.Pi_i_network = nn.Sequential(
            nn.Linear(16, 8), nn.ReLU(), nn.Linear(8, 1), nn.Softplus()
        )

        # =====================
        # SURPRISE ACCUMULATOR
        # =====================
        self.surprise_rnn = nn.GRUCell(input_size=2, hidden_size=16)  # Precision-weighted errors

        # =====================
        # THRESHOLD NETWORK
        # =====================
        # Learns adaptive threshold from metabolic/context signals
        self.threshold_network = nn.Sequential(
            nn.Linear(config.get("context_dim", 8), 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid(),  # Bounded threshold
        )

        # =====================
        # GLOBAL WORKSPACE
        # =====================
        # Gated broadcast layer
        self.workspace = nn.Linear(32 + 16, 64)  # Combined pathways
        self.workspace_gate = nn.Linear(16, 1)  # From surprise accumulator

        # =====================
        # SOMATIC MARKER MODULE
        # =====================
        self.somatic_network = nn.Sequential(
            nn.Linear(64 + config["action_dim"], 32),
            nn.ReLU(),
            nn.Linear(32, config["action_dim"]),  # Value for each action
        )

        # Projection layer to map somatic values to workspace dimension
        self.somatic_projection = nn.Linear(config["action_dim"], 64)

        # =====================
        # OUTPUT HEADS
        # =====================
        self.policy_head = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, config["action_dim"]),
            nn.Softmax(dim=-1),
        )

        self.value_head = nn.Sequential(nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 1))

        # Learnable parameters
        self.beta = nn.Parameter(torch.tensor(1.2))  # Somatic bias
        self.alpha = nn.Parameter(torch.tensor(5.0))  # Sigmoid steepness

        # State
        self.surprise_hidden = None

    def forward(
        self,
        extero_input: torch.Tensor,
        intero_input: torch.Tensor,
        context: torch.Tensor,
        prev_action: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass with APGI dynamics
        """
        batch_size = extero_input.shape[0]

        # Initialize hidden state if needed
        if self.surprise_hidden is None:
            self.surprise_hidden = torch.zeros(batch_size, 16)

        # =====================
        # 1. ENCODE PATHWAYS
        # =====================
        extero_enc = self.extero_encoder(extero_input)  # (B, 32)
        intero_enc = self.intero_encoder(intero_input)  # (B, 16)

        # =====================
        # 2. ESTIMATE PRECISION
        # =====================
        Pi_e = self.Pi_e_network(extero_enc)  # (B, 1)
        Pi_i = self.Pi_i_network(intero_enc)  # (B, 1)

        # =====================
        # 3. COMPUTE PREDICTION ERRORS
        # =====================
        # In practice, these come from comparing predictions to inputs
        # Here simplified as magnitude of encoded signals
        eps_e = torch.norm(extero_enc, dim=-1, keepdim=True)
        eps_i = torch.norm(intero_enc, dim=-1, keepdim=True)

        # =====================
        # 4. PRECISION-WEIGHTED SURPRISE
        # =====================
        weighted_extero = Pi_e * eps_e
        weighted_intero = self.beta * Pi_i * eps_i

        surprise_input = torch.cat([weighted_extero, weighted_intero], dim=-1)

        # Update surprise accumulator
        self.surprise_hidden = self.surprise_rnn(surprise_input, self.surprise_hidden)

        S_t = torch.norm(self.surprise_hidden, dim=-1, keepdim=True)

        # =====================
        # 5. COMPUTE THRESHOLD
        # =====================
        theta_t = self.threshold_network(context)

        # =====================
        # 6. IGNITION GATE
        # =====================
        # Soft gating with learned steepness
        gate_logit = self.alpha * (S_t - theta_t)
        ignition_prob = torch.sigmoid(gate_logit)

        # =====================
        # 7. GLOBAL WORKSPACE
        # =====================
        combined = torch.cat([extero_enc, intero_enc], dim=-1)
        workspace_content = self.workspace(combined)

        # Gated output
        gated_workspace = ignition_prob * workspace_content

        # =====================
        # 8. SOMATIC MARKERS
        # =====================
        if prev_action is not None:
            action_onehot = F.one_hot(prev_action, num_classes=self.config["action_dim"]).float()
            somatic_input = torch.cat([gated_workspace, action_onehot], dim=-1)
            somatic_values = self.somatic_network(somatic_input)
        else:
            somatic_values = torch.zeros(batch_size, self.config["action_dim"])

        # =====================
        # 9. POLICY AND VALUE
        # =====================
        # Combine gated workspace with somatic values for policy
        somatic_projection = self.somatic_projection(somatic_values)
        policy_input = gated_workspace + 0.3 * somatic_projection
        policy = self.policy_head(policy_input)
        value = self.value_head(gated_workspace)

        return {
            "policy": policy,
            "value": value,
            "ignition_prob": ignition_prob,
            "S_t": S_t,
            "theta_t": theta_t,
            "Pi_e": Pi_e,
            "Pi_i": Pi_i,
            "somatic_values": somatic_values,
            "workspace": gated_workspace,
        }

    def reset(self):
        """Reset hidden state"""
        self.surprise_hidden = None


class ComparisonNetworks:
    """Comparison architectures without APGI constraints"""

    @staticmethod
    def create_standard_mlp(config: Dict) -> nn.Module:
        """Standard feedforward network"""

        class StandardMLP(nn.Module):
            def __init__(self, config):
                super().__init__()
                self.network = nn.Sequential(
                    nn.Linear(config["extero_dim"] + config["intero_dim"], 128),
                    nn.ReLU(),
                    nn.Linear(128, 64),
                    nn.ReLU(),
                    nn.Linear(64, config["action_dim"]),
                    nn.Softmax(dim=-1),
                )

            def forward(self, extero, intero, context, prev_action=None):
                x = torch.cat([extero, intero], dim=-1)
                policy = self.network(x)
                return {"policy": policy}

        return StandardMLP(config)

    @staticmethod
    def create_lstm_network(config: Dict) -> nn.Module:
        """Standard LSTM without APGI structure"""

        class LSTMPolicy(nn.Module):
            def __init__(self, config):
                super().__init__()
                input_dim = config["extero_dim"] + config["intero_dim"]
                self.lstm = nn.LSTM(input_dim, 64, batch_first=True)
                self.policy = nn.Linear(64, config["action_dim"])

            def forward(self, extero, intero, context, prev_action=None):
                x = torch.cat([extero, intero], dim=-1).unsqueeze(1)
                lstm_out, _ = self.lstm(x)
                policy = F.softmax(self.policy(lstm_out[:, -1]), dim=-1)
                return {"policy": policy}

        return LSTMPolicy(config)

    @staticmethod
    def create_attention_network(config: Dict) -> nn.Module:
        """Attention-based without explicit ignition"""

        class AttentionPolicy(nn.Module):
            def __init__(self, config):
                super().__init__()
                self.extero_enc = nn.Linear(config["extero_dim"], 32)
                self.intero_enc = nn.Linear(config["intero_dim"], 32)
                self.attention = nn.MultiheadAttention(32, 4)
                self.policy = nn.Linear(32, config["action_dim"])

            def forward(self, extero, intero, context, prev_action=None):
                e = self.extero_enc(extero).unsqueeze(0)
                i = self.intero_enc(intero).unsqueeze(0)
                combined = torch.cat([e, i], dim=0)
                attn_out, _ = self.attention(combined, combined, combined)
                policy = F.softmax(self.policy(attn_out.mean(0)), dim=-1)
                return {"policy": policy}

        return AttentionPolicy(config)


class NetworkComparisonExperiment:
    """Compare APGI-inspired vs standard architectures"""

    def __init__(self, config: Dict):
        # Ensure required dimensions are present
        if "extero_dim" not in config:
            config["extero_dim"] = 32
        if "intero_dim" not in config:
            config["intero_dim"] = 16
        if "action_dim" not in config:
            config["action_dim"] = 4

        self.config = config

        self.networks = {
            "APGI": APGIInspiredNetwork(config),
            "MLP": ComparisonNetworks.create_standard_mlp(config),
            "LSTM": ComparisonNetworks.create_lstm_network(config),
            "Attention": ComparisonNetworks.create_attention_network(config),
        }

    def train_all(self, train_loader, val_loader, n_epochs: int = 100):
        """Train all networks"""
        results = {}

        for name, network in self.networks.items():
            print(f"Training {name}...")

            optimizer = torch.optim.Adam(network.parameters(), lr=1e-3)

            train_losses = []
            val_accuracies = []

            for epoch in range(n_epochs):
                # Training
                network.train()
                epoch_loss = 0

                for batch in train_loader:
                    optimizer.zero_grad()

                    outputs = network(
                        batch["extero"],
                        batch["intero"],
                        batch["context"],
                        batch.get("prev_action"),
                    )

                    # Policy loss (cross-entropy with target action)
                    loss = F.cross_entropy(outputs["policy"], batch["target_action"])

                    # Add auxiliary losses for APGI network
                    if name == "APGI" and "target_ignition" in batch:
                        ignition_loss = F.binary_cross_entropy(
                            outputs["ignition_prob"].squeeze(),
                            batch["target_ignition"].float(),
                        )
                        loss += 0.1 * ignition_loss

                    loss.backward()
                    optimizer.step()

                    epoch_loss += loss.item()

                train_losses.append(epoch_loss / len(train_loader))

                # Validation
                network.eval()
                correct = 0
                total = 0

                with torch.no_grad():
                    for batch in val_loader:
                        outputs = network(batch["extero"], batch["intero"], batch["context"])

                        pred = outputs["policy"].argmax(dim=-1)
                        correct += (pred == batch["target_action"]).sum().item()
                        total += len(pred)

                val_accuracies.append(correct / total)

            results[name] = {
                "train_losses": train_losses,
                "val_accuracies": val_accuracies,
                "final_accuracy": val_accuracies[-1],
            }

        return results

    def evaluate_on_tasks(self, task_datasets: Dict) -> Dict:
        """
        Evaluate on consciousness-relevant tasks

        Tasks:
        1. Conscious/unconscious classification
        2. Masking threshold detection
        3. Attentional blink prediction
        4. Interoceptive accuracy
        """
        task_results = {}

        for task_name, dataset in task_datasets.items():
            task_results[task_name] = {}

            for net_name, network in self.networks.items():
                network.eval()

                predictions = []
                targets = []

                with torch.no_grad():
                    for batch in dataset:
                        outputs = network(batch["extero"], batch["intero"], batch["context"])

                        if task_name == "conscious_classification":
                            # Use ignition probability as prediction
                            if "ignition_prob" in outputs:
                                pred = outputs["ignition_prob"].squeeze()
                            else:
                                # For non-APGI networks, use policy entropy
                                policy = outputs["policy"]
                                entropy = -(policy * torch.log(policy + 1e-10)).sum(-1)
                                pred = 1 - entropy / np.log(policy.shape[-1])
                        else:
                            pred = outputs["policy"].argmax(dim=-1)

                        predictions.append(pred)
                        targets.append(batch["target"])

                predictions = torch.cat(predictions)
                targets = torch.cat(targets)

                # Compute metrics
                if task_name == "conscious_classification":
                    from sklearn.metrics import roc_auc_score

                    auc = roc_auc_score(targets.numpy(), predictions.numpy())
                    task_results[task_name][net_name] = {"auc": auc}
                else:
                    accuracy = (predictions == targets).float().mean().item()
                    task_results[task_name][net_name] = {"accuracy": accuracy}

        return task_results

    def run_full_experiment(self) -> Dict:
        """Run a complete comparison experiment with synthetic data"""
        print("Starting Network Comparison Experiment...")

        # Create synthetic datasets for evaluation
        task_datasets = {}

        # Task 1: Conscious/unconscious classification
        n_samples = 200
        task_datasets["conscious_classification"] = []
        for _ in range(5):  # 5 batches
            batch_size = n_samples // 5
            extero = torch.randn(batch_size, self.config["extero_dim"])
            intero = torch.randn(batch_size, self.config["intero_dim"])
            context = torch.randn(batch_size, self.config.get("context_dim", 8))

            # Target: high surprise = conscious (1), low surprise = unconscious (0)
            target = (torch.norm(extero, dim=1) > 1.0).float()

            # For APGI network, also provide ignition targets
            target_ignition = target

            batch = {
                "extero": extero,
                "intero": intero,
                "context": context,
                "target": target,
                "target_ignition": target_ignition,
            }
            task_datasets["conscious_classification"].append(batch)

        # Task 2: Simple action selection
        task_datasets["action_selection"] = []
        for _ in range(5):
            batch_size = n_samples // 5
            extero = torch.randn(batch_size, self.config["extero_dim"])
            intero = torch.randn(batch_size, self.config["intero_dim"])
            context = torch.randn(batch_size, self.config.get("context_dim", 8))
            target_action = torch.randint(0, self.config["action_dim"], (batch_size,))

            batch = {
                "extero": extero,
                "intero": intero,
                "context": context,
                "target": target_action,
            }
            task_datasets["action_selection"].append(batch)

        # Evaluate all networks
        results = self.evaluate_on_tasks(task_datasets)

        print("Network Comparison Results:")
        for task_name, task_results in results.items():
            print(f"\n{task_name}:")
            for net_name, metrics in task_results.items():
                if "auc" in metrics:
                    print(f"  {net_name}: AUC = {metrics['auc']:.3f}")
                else:
                    print(f"  {net_name}: Accuracy = {metrics['accuracy']:.3f}")

        return results


# Main execution
if __name__ == "__main__":
    print("Starting Network Comparison Experiment...")

    # Default configuration
    config = {"extero_dim": 32, "intero_dim": 16, "action_dim": 4, "context_dim": 8}

    experiment = NetworkComparisonExperiment(config)
    results = experiment.run_full_experiment()

    print("=== Protocol completed successfully ===")


def run_falsification():
    """Entry point for CLI falsification testing."""
    try:
        print("Running APGI Falsification Protocol 6: Network Comparison Experiment")
        config = {
            "extero_dim": 32,
            "intero_dim": 16,
            "action_dim": 4,
            "context_dim": 8,
        }

        experiment = NetworkComparisonExperiment(config)
        results = experiment.run_full_experiment()

        print("=== Protocol completed successfully ===")
        return {"status": "success", "results": results}
    except (RuntimeError, ValueError, TypeError, ImportError, KeyError) as e:
        print(f"Error in falsification protocol 6: {e}")
        return {"status": "error", "message": str(e)}
