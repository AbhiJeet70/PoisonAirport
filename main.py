   import torch
   from torch_geometric.datasets import Airports
   from torch_geometric.transforms import NormalizeFeatures
   from torch_geometric.nn import GCNConv
   from gcn_model import GCNNet
   from utils import load_airports_data, split_indices, print_dataset_statistics, find_extreme_degree_nodes, perturb_edges, train_model
   import matplotlib.pyplot as plt

   # Set random seed for reproducibility

  def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

set_seed(20)

   # List of countries to process
   countries = ['USA', 'Brazil', 'Europe']

   # Process each country and print accuracies
   for country in countries:
       print(f'Processing country: {country}')
       data = load_airports_data(country)
       
       # Print dataset statistics
       print_dataset_statistics(data, country)
       
       # Prepare the masks
       train_idx, val_idx, test_idx = split_indices(data.num_nodes)
       data.train_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
       data.val_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
       data.test_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
       data.train_mask[train_idx] = True
       data.val_mask[val_idx] = True
       data.test_mask[test_idx] = True
       
       best_acc = 0
       best_params = None
       
       models = [GCNNet(data.num_node_features, 256, data.y.max().item() + 1) for _ in range(3)]
       for model in models:
           for lr in [0.01, 0.001]:
               for weight_decay in [1e-4, 1e-5]:
                   print(f'Training with {model.__class__.__name__}, lr={lr}, weight_decay={weight_decay}')
                   clean_acc = train_model(model, data, lr, weight_decay)
                   if clean_acc > best_acc:
                       best_acc = clean_acc
                       best_params = (model.__class__.__name__, model, lr, weight_decay)
       
       print(f'Clean accuracy for {country}: {best_acc:.4f} with params {best_params}')
       
       # Store accuracies for plotting
       perturbation_percentages = [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.30, 0.35, 0.4, 0.45, 0.5]
       accuracies = [best_acc]
       
       for perturbation_percentage in perturbation_percentages[1:]:
           # Load data again to avoid perturbing already perturbed data
           data = load_airports_data(country)
           
           # Prepare the masks again
           data.train_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
           data.val_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
           data.test_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
           data.train_mask[train_idx] = True
           data.val_mask[val_idx] = True
           data.test_mask[test_idx] = True
           
           # Perturb edges of top degree nodes
           data = perturb_edges(data, find_extreme_degree_nodes(data.edge_index, data.num_nodes)[0], perturbation_percentage=perturbation_percentage)
           
           worst_acc = 1
           
           for model in models:
               print(f'Training with {model.__class__.__name__}, lr={best_params[2]}, weight_decay={best_params[3]} for perturbation {perturbation_percentage*100}%')
               perturbed_acc = train_model(model, data, best_params[2], best_params[3])
               if perturbed_acc < worst_acc:
                   worst_acc = perturbed_acc
           
           print(f'Worst accuracy after perturbing {perturbation_percentage*100}% edges for {country}: {worst_acc:.4f}')
           accuracies.append(worst_acc)
       
       # Plot accuracies
       plt.figure(figsize=(10, 6))
       plt.plot([0] + [p * 100 for p in perturbation_percentages[1:]], accuracies, marker='o')
       plt.xlabel('Perturbation Percentage')
       plt.ylabel('Accuracy')
       plt.title(f'Accuracy vs Perturbation for {country}')
       plt.grid(True)
       plt.show()
