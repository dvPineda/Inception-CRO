# src/research_metrics.py

import numpy as np
import torch
import time
import psutil
import os
from typing import Dict, List, Tuple, Any, Optional
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score,
    average_precision_score, matthews_corrcoef
)
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

class ResearchMetrics:
    """
    Comprehensive metrics collection for research-level model evaluation.
    Includes performance, efficiency, statistical, and computational metrics.
    """
    
    def __init__(self, device: torch.device):
        self.device = device
        self.metrics = {}
        
    def evaluate_comprehensive(
        self, 
        model: torch.nn.Module, 
        test_loader: torch.utils.data.DataLoader,
        val_loader: torch.utils.data.DataLoader = None,
        model_name: str = "model",
        num_runs: int = 3
    ) -> Dict[str, Any]:
        """
        Perform comprehensive research-level evaluation.
        
        Args:
            model: PyTorch model to evaluate
            test_loader: Test data loader
            val_loader: Validation data loader (optional)
            model_name: Name identifier for the model
            num_runs: Number of evaluation runs for statistical significance
            
        Returns:
            Dictionary with comprehensive metrics
        """
        print(f"\n🔬 Research-level evaluation of {model_name}...")
        
        results = {
            'model_name': model_name,
            'evaluation_runs': num_runs
        }
        
        # 1. Performance Metrics (multiple runs for statistical significance)
        print("📊 Computing performance metrics...")
        performance_results = self._evaluate_performance_multirun(
            model, test_loader, num_runs
        )
        results.update(performance_results)
        
        # 2. Model Complexity Metrics
        print("🔢 Analyzing model complexity...")
        complexity_results = self._analyze_model_complexity(model)
        results.update(complexity_results)
        
        # 3. Computational Efficiency Metrics
        print("⚡ Measuring computational efficiency...")
        efficiency_results = self._measure_computational_efficiency(
            model, test_loader
        )
        results.update(efficiency_results)
        
        # 4. Memory Usage Analysis
        print("💾 Analyzing memory usage...")
        memory_results = self._analyze_memory_usage(model, test_loader)
        results.update(memory_results)
        
        # 5. Robustness Metrics (if validation set available)
        if val_loader is not None:
            print("🛡️ Computing robustness metrics...")
            robustness_results = self._compute_robustness_metrics(
                model, test_loader, val_loader
            )
            results.update(robustness_results)
        
        # 6. Green AI Metrics
        print("🌱 Computing Green AI metrics...")
        green_metrics = self._compute_green_ai_metrics(results)
        results.update(green_metrics)
        
        # 7. Statistical Significance Tests
        print("📈 Computing statistical measures...")
        statistical_results = self._compute_statistical_measures(results)
        results.update(statistical_results)
        
        return results
    
    def _evaluate_performance_multirun(
        self, 
        model: torch.nn.Module, 
        test_loader: torch.utils.data.DataLoader, 
        num_runs: int
    ) -> Dict[str, Any]:
        """Evaluate model performance across multiple runs for statistical significance."""
        
        all_predictions = []
        all_true_labels = []
        all_probabilities = []
        run_accuracies = []
        run_times = []
        
        model.eval()
        
        for run in range(num_runs):
            print(f"  Run {run + 1}/{num_runs}")
            
            predictions = []
            true_labels = []
            probabilities = []
            
            start_time = time.time()
            
            with torch.no_grad():
                for batch_idx, (data, target) in enumerate(test_loader):
                    data, target = data.to(self.device), target.to(self.device)
                    
                    # Handle multi-label vs multi-class
                    if len(target.shape) > 1 and target.shape[1] > 1:
                        # One-hot encoded targets - convert to class indices for compatibility
                        target_classes = target.argmax(dim=1)  # Convert to class indices
                        
                        # Multi-class case with converted targets
                        output = model(data)
                        prob = torch.softmax(output, dim=1)
                        pred = output.argmax(dim=1, keepdim=True)
                        target = target_classes.unsqueeze(1)  # Match prediction shape
                    else:
                        # Standard multi-class case
                        output = model(data)
                        prob = torch.softmax(output, dim=1)
                        pred = output.argmax(dim=1, keepdim=True)
                        if len(target.shape) > 1:
                            target = target.squeeze()
                    
                    predictions.append(pred.cpu().numpy())
                    true_labels.append(target.cpu().numpy())
                    probabilities.append(prob.cpu().numpy())
            
            run_time = time.time() - start_time
            run_times.append(run_time)
            
            # Concatenate batch results
            run_preds = np.concatenate(predictions, axis=0)
            run_labels = np.concatenate(true_labels, axis=0)
            run_probs = np.concatenate(probabilities, axis=0)
            
            # Calculate accuracy for this run
            if len(run_labels.shape) > 1 and run_labels.shape[1] > 1:
                # Multi-label accuracy
                accuracy = np.mean(run_preds == run_labels)
            else:
                # Multi-class accuracy
                accuracy = accuracy_score(run_labels.flatten(), run_preds.flatten())
            
            run_accuracies.append(accuracy)
            
            # Store for ensemble analysis
            if run == 0:  # Use first run for detailed metrics
                all_predictions = run_preds
                all_true_labels = run_labels
                all_probabilities = run_probs
        
        # Compute detailed metrics on first run
        detailed_metrics = self._compute_detailed_classification_metrics(
            all_true_labels, all_predictions, all_probabilities
        )
        
        # Statistical analysis across runs
        accuracy_stats = {
            'accuracy_mean': np.mean(run_accuracies),
            'accuracy_std': np.std(run_accuracies),
            'accuracy_min': np.min(run_accuracies),
            'accuracy_max': np.max(run_accuracies),
            'accuracy_ci_95': self._compute_confidence_interval(run_accuracies, 0.95),
            'inference_time_mean': np.mean(run_times),
            'inference_time_std': np.std(run_times)
        }
        
        return {**detailed_metrics, **accuracy_stats}
    
    def _compute_detailed_classification_metrics(
        self, 
        y_true: np.ndarray, 
        y_pred: np.ndarray, 
        y_prob: np.ndarray
    ) -> Dict[str, Any]:
        """Compute comprehensive classification metrics."""
        
        metrics = {}
        
        # Determine if multi-label or multi-class
        is_multilabel = len(y_true.shape) > 1 and y_true.shape[1] > 1
        
        if is_multilabel:
            # Multi-label metrics
            metrics.update({
                'accuracy': np.mean(y_pred == y_true),
                'hamming_loss': np.mean(y_pred != y_true),
                'exact_match_ratio': np.mean(np.all(y_pred == y_true, axis=1)),
                'precision_macro': precision_score(y_true, y_pred, average='macro', zero_division=0),
                'recall_macro': recall_score(y_true, y_pred, average='macro', zero_division=0),
                'f1_macro': f1_score(y_true, y_pred, average='macro', zero_division=0),
                'precision_micro': precision_score(y_true, y_pred, average='micro', zero_division=0),
                'recall_micro': recall_score(y_true, y_pred, average='micro', zero_division=0),
                'f1_micro': f1_score(y_true, y_pred, average='micro', zero_division=0)
            })
            
            # Try to compute AUC for multi-label
            try:
                metrics['auc_macro'] = roc_auc_score(y_true, y_prob, average='macro')
                metrics['auc_micro'] = roc_auc_score(y_true, y_prob, average='micro')
            except:
                metrics['auc_macro'] = 0.0
                metrics['auc_micro'] = 0.0
                
        else:
            # Multi-class metrics
            y_true_flat = y_true.flatten()
            y_pred_flat = y_pred.flatten()
            
            metrics.update({
                'accuracy': accuracy_score(y_true_flat, y_pred_flat),
                'precision_macro': precision_score(y_true_flat, y_pred_flat, average='macro', zero_division=0),
                'recall_macro': recall_score(y_true_flat, y_pred_flat, average='macro', zero_division=0),
                'f1_macro': f1_score(y_true_flat, y_pred_flat, average='macro', zero_division=0),
                'precision_weighted': precision_score(y_true_flat, y_pred_flat, average='weighted', zero_division=0),
                'recall_weighted': recall_score(y_true_flat, y_pred_flat, average='weighted', zero_division=0),
                'f1_weighted': f1_score(y_true_flat, y_pred_flat, average='weighted', zero_division=0),
                'matthews_corrcoef': matthews_corrcoef(y_true_flat, y_pred_flat)
            })
            
            # Confusion matrix analysis
            cm = confusion_matrix(y_true_flat, y_pred_flat)
            metrics.update({
                'confusion_matrix': cm.tolist(),
                'num_classes': len(np.unique(y_true_flat))
            })
            
            # AUC metrics (for multi-class)
            try:
                if y_prob.shape[1] > 2:  # Multi-class
                    metrics['auc_ovr'] = roc_auc_score(y_true_flat, y_prob, multi_class='ovr', average='macro')
                    metrics['auc_ovo'] = roc_auc_score(y_true_flat, y_prob, multi_class='ovo', average='macro')
                else:  # Binary
                    metrics['auc'] = roc_auc_score(y_true_flat, y_prob[:, 1])
            except:
                pass
        
        return metrics
    
    def _analyze_model_complexity(self, model: torch.nn.Module) -> Dict[str, Any]:
        """Analyze model architectural complexity."""
        
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        # Layer analysis
        layer_types = {}
        layer_params = {}
        
        for name, module in model.named_modules():
            module_type = type(module).__name__
            if module_type not in layer_types:
                layer_types[module_type] = 0
                layer_params[module_type] = 0
            
            layer_types[module_type] += 1
            layer_params[module_type] += sum(p.numel() for p in module.parameters())
        
        # Model depth analysis
        def count_depth(module, depth=0):
            max_depth = depth
            for child in module.children():
                child_depth = count_depth(child, depth + 1)
                max_depth = max(max_depth, child_depth)
            return max_depth
        
        model_depth = count_depth(model)
        
        return {
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'non_trainable_parameters': total_params - trainable_params,
            'model_depth': model_depth,
            'layer_types': layer_types,
            'parameters_per_layer_type': layer_params,
            'parameters_mb': total_params * 4 / (1024 * 1024),  # Assuming float32
            'model_size_mb': self._estimate_model_size_mb(model)
        }
    
    def _measure_computational_efficiency(self, model: torch.nn.Module, test_loader: torch.utils.data.DataLoader) -> Dict[str, Any]:
        """Measure computational efficiency metrics."""
        
        model.eval()
        
        # Warmup
        with torch.no_grad():
            for i, (data, _) in enumerate(test_loader):
                if i >= 3:  # Warmup with 3 batches
                    break
                data = data.to(self.device)
                _ = model(data)
        
        # Timing measurements
        batch_times = []
        throughput_samples = []
        
        torch.cuda.synchronize() if self.device.type == 'cuda' else None
        
        with torch.no_grad():
            for i, (data, _) in enumerate(test_loader):
                if i >= 50:  # Test on 50 batches for statistical significance
                    break
                    
                data = data.to(self.device)
                batch_size = data.size(0)
                
                start_time = time.perf_counter()
                _ = model(data)
                torch.cuda.synchronize() if self.device.type == 'cuda' else None
                end_time = time.perf_counter()
                
                batch_time = end_time - start_time
                batch_times.append(batch_time)
                throughput_samples.append(batch_size / batch_time)
        
        # FLOPs estimation (approximate)
        flops = self._estimate_flops(model, next(iter(test_loader))[0][:1])
        
        return {
            'inference_time_per_batch_ms': np.mean(batch_times) * 1000,
            'inference_time_std_ms': np.std(batch_times) * 1000,
            'throughput_samples_per_sec': np.mean(throughput_samples),
            'throughput_std': np.std(throughput_samples),
            'estimated_flops': flops,
            'flops_per_parameter': flops / sum(p.numel() for p in model.parameters()) if flops > 0 else 0
        }
    
    def _analyze_memory_usage(self, model: torch.nn.Module, test_loader: torch.utils.data.DataLoader) -> Dict[str, Any]:
        """Analyze memory usage patterns."""
        
        # Model memory
        model_memory_mb = sum(p.numel() * p.element_size() for p in model.parameters()) / (1024 * 1024)
        
        # Runtime memory measurement
        if self.device.type == 'cuda':
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            
            # Measure peak memory during inference
            model.eval()
            with torch.no_grad():
                data, _ = next(iter(test_loader))
                data = data.to(self.device)
                _ = model(data)
                
            peak_memory_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)
            current_memory_mb = torch.cuda.memory_allocated() / (1024 * 1024)
            
            memory_metrics = {
                'model_memory_mb': model_memory_mb,
                'peak_gpu_memory_mb': peak_memory_mb,
                'current_gpu_memory_mb': current_memory_mb,
                'memory_efficiency': model_memory_mb / peak_memory_mb if peak_memory_mb > 0 else 0
            }
        else:
            # CPU memory measurement
            process = psutil.Process(os.getpid())
            memory_info = process.memory_info()
            
            memory_metrics = {
                'model_memory_mb': model_memory_mb,
                'cpu_memory_mb': memory_info.rss / (1024 * 1024),
                'memory_efficiency': 1.0  # Placeholder for CPU
            }
        
        return memory_metrics
    
    def _compute_robustness_metrics(self, model: torch.nn.Module, test_loader: torch.utils.data.DataLoader, val_loader: torch.utils.data.DataLoader) -> Dict[str, Any]:
        """Compute model robustness and generalization metrics."""
        
        # Evaluate on both validation and test sets
        val_results = self._evaluate_performance_multirun(model, val_loader, 1)
        test_results = self._evaluate_performance_multirun(model, test_loader, 1)
        
        # Generalization gap
        generalization_gap = val_results['accuracy_mean'] - test_results['accuracy_mean']
        
        # Stability across different data splits
        stability_metrics = {
            'generalization_gap': generalization_gap,
            'val_test_accuracy_ratio': test_results['accuracy_mean'] / val_results['accuracy_mean'] if val_results['accuracy_mean'] > 0 else 0,
            'performance_consistency': 1.0 - abs(generalization_gap)  # Higher is better
        }
        
        return stability_metrics
    
    def _compute_green_ai_metrics(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Compute Green AI and sustainability metrics."""
        
        # Energy efficiency proxies
        params = results.get('total_parameters', 1)
        flops = results.get('estimated_flops', 0)
        inference_time = results.get('inference_time_per_batch_ms', 0)
        accuracy = results.get('accuracy_mean', 0)
        
        green_metrics = {
            'parameter_efficiency': accuracy / (params / 1e6) if params > 0 else 0,  # Accuracy per million parameters
            'flops_efficiency': accuracy / (flops / 1e9) if flops > 0 else 0,  # Accuracy per GFLOP
            'time_efficiency': accuracy / (inference_time / 1000) if inference_time > 0 else 0,  # Accuracy per second
            'sustainability_score': self._compute_sustainability_score(results),
            'carbon_footprint_relative': self._estimate_carbon_footprint(results)
        }
        
        return green_metrics
    
    def _compute_statistical_measures(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Compute statistical significance and confidence measures."""
        
        accuracy_mean = results.get('accuracy_mean', 0)
        accuracy_std = results.get('accuracy_std', 0)
        
        # Statistical measures
        statistical_metrics = {
            'coefficient_of_variation': accuracy_std / accuracy_mean if accuracy_mean > 0 else 0,
            'reliability_score': max(0, 1 - 2 * accuracy_std),  # Higher is better
            'statistical_power': self._estimate_statistical_power(results)
        }
        
        return statistical_metrics
    
    # Helper methods
    def _compute_confidence_interval(self, data: List[float], confidence: float) -> Tuple[float, float]:
        """Compute confidence interval for given data."""
        data = np.array(data)
        n = len(data)
        mean = np.mean(data)
        sem = stats.sem(data)
        h = sem * stats.t.ppf((1 + confidence) / 2., n-1)
        return (mean - h, mean + h)
    
    def _estimate_model_size_mb(self, model: torch.nn.Module) -> float:
        """Estimate model size in MB."""
        param_size = 0
        buffer_size = 0
        
        for param in model.parameters():
            param_size += param.nelement() * param.element_size()
        
        for buffer in model.buffers():
            buffer_size += buffer.nelement() * buffer.element_size()
        
        return (param_size + buffer_size) / (1024 * 1024)
    
    def _estimate_flops(self, model: torch.nn.Module, input_tensor: torch.Tensor) -> int:
        """Estimate FLOPs for the model (simplified estimation)."""
        try:
            # This is a simplified FLOP estimation
            # In practice, you might want to use libraries like thop or ptflops
            total_flops = 0
            
            def flop_count_hook(module, input, output):
                nonlocal total_flops
                if isinstance(module, torch.nn.Conv2d):
                    # Conv2d FLOPs = output_elements * (kernel_size * in_channels + bias)
                    output_elements = output.numel()
                    kernel_flops = module.kernel_size[0] * module.kernel_size[1] * module.in_channels
                    total_flops += output_elements * kernel_flops
                elif isinstance(module, torch.nn.Linear):
                    # Linear FLOPs = output_elements * in_features
                    total_flops += output.numel() * module.in_features
            
            hooks = []
            for module in model.modules():
                if isinstance(module, (torch.nn.Conv2d, torch.nn.Linear)):
                    hooks.append(module.register_forward_hook(flop_count_hook))
            
            model.eval()
            with torch.no_grad():
                _ = model(input_tensor.to(self.device))
            
            for hook in hooks:
                hook.remove()
                
            return total_flops
        except:
            return 0
    
    def _compute_sustainability_score(self, results: Dict[str, Any]) -> float:
        """Compute a composite sustainability score."""
        # Normalize and combine different efficiency metrics
        param_eff = results.get('parameter_efficiency', 0)
        time_eff = results.get('time_efficiency', 0)
        
        # Simple weighted combination (can be made more sophisticated)
        sustainability = 0.6 * min(param_eff / 100, 1.0) + 0.4 * min(time_eff / 1000, 1.0)
        return sustainability
    
    def _estimate_carbon_footprint(self, results: Dict[str, Any]) -> float:
        """Estimate relative carbon footprint (proxy)."""
        # Very simplified estimation based on computational requirements
        params = results.get('total_parameters', 1)
        flops = results.get('estimated_flops', 0)
        
        # Relative footprint (normalized to [0,1] range)
        footprint = (params / 1e7 + flops / 1e10) / 2
        return min(footprint, 1.0)
    
    def _estimate_statistical_power(self, results: Dict[str, Any]) -> float:
        """Estimate statistical power of the evaluation."""
        # Simplified statistical power estimation
        accuracy_std = results.get('accuracy_std', 0)
        num_runs = results.get('evaluation_runs', 1)
        
        # Higher number of runs and lower variance = higher power
        power = min(1.0, (num_runs / 10) * (1 - accuracy_std * 10))
        return max(0.0, power)


def compare_models_statistical_significance(
    results1: Dict[str, Any], 
    results2: Dict[str, Any], 
    alpha: float = 0.05
) -> Dict[str, Any]:
    """
    Perform statistical significance testing between two models.
    
    Args:
        results1: Results from first model
        results2: Results from second model  
        alpha: Significance level
        
    Returns:
        Statistical comparison results
    """
    
    # Extract accuracy values (would need to be stored from multiple runs)
    acc1_mean = results1.get('accuracy_mean', 0)
    acc1_std = results1.get('accuracy_std', 0)
    acc2_mean = results2.get('accuracy_mean', 0)
    acc2_std = results2.get('accuracy_std', 0)
    
    # Simple statistical tests (in practice, you'd use the raw data)
    # This is a simplified version - you'd want the actual run data
    
    statistical_comparison = {
        'model1_name': results1.get('model_name', 'Model1'),
        'model2_name': results2.get('model_name', 'Model2'),
        'accuracy_difference': acc1_mean - acc2_mean,
        'effect_size': abs(acc1_mean - acc2_mean) / np.sqrt((acc1_std**2 + acc2_std**2) / 2),
        'practical_significance': abs(acc1_mean - acc2_mean) > 0.01,  # 1% threshold
        'confidence_level': 1 - alpha
    }
    
    return statistical_comparison

