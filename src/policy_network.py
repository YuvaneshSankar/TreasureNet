import torch
import torch.nn as nn
import torch.nn.functional as F

class PolicyNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.001):
        super(PolicyNetwork, self).__init__()
        
        # Two hidden layers - perfect for grid world spatial relationships
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)
        
        self.learning_rate = learning_rate
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        
    def forward(self, x):
        # If batch size =1 then this handles it 
        if len(x.shape) == 1:
            x = x.unsqueeze(0)
            
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        
        action_probs = F.softmax(x, dim=-1)
        return action_probs
    

    #To calculate the policy gradient for updating weights like log(pi(a|s)->action_probs)
    def get_log_probs(self, x):
        # If batch size =1 then this handles it 
        if len(x.shape) == 1:
            x = x.unsqueeze(0)
            
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        
        log_probs = F.log_softmax(x, dim=-1)
        return log_probs
    
    def update_weights(self, loss):
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()
    
    def set_learning_rate(self, new_lr):
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = new_lr

# Quick test when running this file
if __name__ == "__main__":
    print("Testing PolicyNetwork...")
    
    # 2 inputs (x,y), 64 hidden units, 4 outputs (actions -> up, down, left, right)
    model = PolicyNetwork(input_size=2, hidden_size=64, output_size=4)
    
    print("Network architecture:")
    print(model)
    print()
    
    # Test 1: Single state
    print("Test 1: Single state")
    single_state = torch.tensor([3.0, 2.0])  # x=3, y=2
    action_probs = model(single_state)
    print(f"Input state: {single_state}")
    print(f"Action probabilities: {action_probs}")
    print(f"Probabilities sum to: {action_probs.sum().item():.6f}")
    print()
    
    # Test 2: Batch of states
    print("Test 2: Batch of states")
    batch_states = torch.tensor([
        [0.0, 0.0],  # Top-left corner
        [2.0, 2.0],  # Middle
        [4.0, 4.0],  # Bottom-right corner
    ])
    batch_probs = model(batch_states)
    print(f"Input batch shape: {batch_states.shape}")
    print(f"Output batch shape: {batch_probs.shape}")
    print(f"Batch probabilities:")
    for i, probs in enumerate(batch_probs):
        print(f"  State {i}: {probs} (sum: {probs.sum().item():.6f})")
    print()
    
    # Test 3: Log probabilities for REINFORCE
    print("Test 3: Log probabilities")
    log_probs = model.get_log_probs(single_state)
    print(f"Log probabilities: {log_probs}")
    print(f"Regular probs from log: {torch.exp(log_probs)}")
    print()
    
    # Test 4: Gradient computation
    print("Test 4: Gradient computation")
    dummy_loss = -log_probs[0, 1]  # Negative log prob of action 1
    print(f"Dummy loss: {dummy_loss.item():.6f}")
    
    # Check if gradients can be computed
    dummy_loss.backward()
    print("Gradients computed successfully!")
    
    # Check sample gradients
    print("Sample gradients:")
    for name, param in model.named_parameters():
        if param.grad is not None:
            print(f"  {name}: grad norm = {param.grad.norm().item():.6f}")
        else:
            print(f"  {name}: No gradient")
    
    print("\n All tests passed! Policy network is ready.")


