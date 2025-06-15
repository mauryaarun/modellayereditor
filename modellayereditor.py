#!/usr/bin/env python3
"""
Model Layer Editor Tool
Provides comprehensive functionality for manipulating transformer models including
layer replacement, analysis, visualization, and validation.
"""

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import warnings

import torch
import torch.nn.functional as F
from transformers import (
    AutoModel, AutoConfig, AutoTokenizer, 
    AutoModelForCausalLM, AutoModelForSequenceClassification,
    logging as transformers_logging
)
from transformers.models.auto.modeling_auto import MODEL_FOR_CAUSAL_LM_MAPPING
import numpy as np

# Suppress transformers warnings for cleaner output
transformers_logging.set_verbosity_error()
warnings.filterwarnings("ignore")


class ModelAnalyzer:
    """Analyzes model architecture and provides insights."""
    
    @staticmethod
    def get_layer_info(model) -> Dict[str, Dict]:
        """Extract detailed information about model layers."""
        layer_info = {}
        state_dict = model.state_dict()
        
        for name, param in state_dict.items():
            layer_parts = name.split('.')
            layer_type = layer_parts[-1] if len(layer_parts) > 1 else 'root'
            
            layer_info[name] = {
                'shape': list(param.shape),
                'dtype': str(param.dtype),
                'size': param.numel(),
                'layer_type': layer_type,
                'frozen': not param.requires_grad if hasattr(param, 'requires_grad') else False
            }
        
        return layer_info
    
    @staticmethod
    def find_compatible_layers(base_info: Dict, donor_info: Dict, 
                             similarity_threshold: float = 0.9) -> List[Tuple[str, str, float]]:
        """Find compatible layers between models based on shape and name similarity."""
        compatible = []
        
        for base_name, base_data in base_info.items():
            for donor_name, donor_data in donor_info.items():
                # Shape compatibility
                shape_match = base_data['shape'] == donor_data['shape']
                
                # Name similarity (simple approach)
                name_similarity = len(set(base_name.split('.')) & set(donor_name.split('.'))) / \
                                max(len(base_name.split('.')), len(donor_name.split('.')))
                
                if shape_match and name_similarity >= similarity_threshold:
                    compatible.append((base_name, donor_name, name_similarity))
        
        return sorted(compatible, key=lambda x: x[2], reverse=True)


class ModelManipulator:
    """Main class for model manipulation operations."""
    
    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if self.verbose:
            print(f"Using device: {self.device}")
    
    def _print(self, message: str):
        """Print message if verbose mode is enabled."""
        if self.verbose:
            print(message)
    
    def load_model(self, model_name: str, model_type: str = 'auto') -> torch.nn.Module:
        """Load model with appropriate class based on type."""
        self._print(f"Loading model: {model_name}")
        
        try:
            if model_type == 'causal':
                model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16)
            elif model_type == 'classification':
                model = AutoModelForSequenceClassification.from_pretrained(model_name)
            else:
                model = AutoModel.from_pretrained(model_name)
            
            return model.to(self.device)
        except Exception as e:
            self._print(f"Error loading model {model_name}: {e}")
            raise
    
    def analyze_models(self, base_model_name: str, donor_model_name: str, 
                      output_file: Optional[str] = None) -> Dict:
        """Analyze and compare two models."""
        self._print("Analyzing models...")
        
        base_model = self.load_model(base_model_name)
        donor_model = self.load_model(donor_model_name)
        
        base_info = ModelAnalyzer.get_layer_info(base_model)
        donor_info = ModelAnalyzer.get_layer_info(donor_model)
        
        compatible_layers = ModelAnalyzer.find_compatible_layers(base_info, donor_info)
        
        analysis = {
            'base_model': {
                'name': base_model_name,
                'total_parameters': sum(p.numel() for p in base_model.parameters()),
                'layer_count': len(base_info),
                'layers': base_info
            },
            'donor_model': {
                'name': donor_model_name,
                'total_parameters': sum(p.numel() for p in donor_model.parameters()),
                'layer_count': len(donor_info),
                'layers': donor_info
            },
            'compatibility': {
                'compatible_layers': len(compatible_layers),
                'matches': compatible_layers[:20]  # Top 20 matches
            }
        }
        
        if output_file:
            with open(output_file, 'w') as f:
                json.dump(analysis, f, indent=2)
            self._print(f"Analysis saved to: {output_file}")
        
        return analysis
    
    def replace_layers(self, base_model_name: str, donor_model_name: str,
                      layers_to_replace: Union[str, List[str]], output_dir: str,
                      model_type: str = 'auto', validate: bool = True,
                      backup: bool = True) -> bool:
        """Replace specified layers with enhanced validation and backup."""
        
        if isinstance(layers_to_replace, str):
            layers_to_replace = [layers_to_replace]
        
        # Create output directory
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        # Load models
        base_model = self.load_model(base_model_name, model_type)
        donor_model = self.load_model(donor_model_name, model_type)
        
        # Get state dictionaries
        base_state_dict = base_model.state_dict()
        donor_state_dict = donor_model.state_dict()
        
        # Backup original model if requested
        if backup:
            backup_path = Path(output_dir) / "original_backup"
            backup_path.mkdir(exist_ok=True)
            base_model.save_pretrained(str(backup_path))
            self._print(f"Original model backed up to: {backup_path}")
        
        # Track replacements
        replacements_made = []
        replacements_failed = []
        
        # Replace layers
        for layer_pattern in layers_to_replace:
            pattern_matches = []
            
            for key in donor_state_dict:
                if re.search(layer_pattern, key) or layer_pattern in key:
                    if key in base_state_dict:
                        # Validate shape compatibility
                        if base_state_dict[key].shape == donor_state_dict[key].shape:
                            self._print(f"Replacing: {key}")
                            base_state_dict[key] = donor_state_dict[key].clone()
                            pattern_matches.append(key)
                            replacements_made.append(key)
                        else:
                            self._print(f"Shape mismatch for {key}: "
                                      f"{base_state_dict[key].shape} vs {donor_state_dict[key].shape}")
                            replacements_failed.append((key, "shape_mismatch"))
                    else:
                        self._print(f"Key not found in base model: {key}")
                        replacements_failed.append((key, "key_not_found"))
            
            if not pattern_matches:
                self._print(f"No matching keys found for pattern: {layer_pattern}")
                replacements_failed.append((layer_pattern, "no_matches"))
        
        if not replacements_made:
            self._print("No replacements were made. Exiting.")
            return False
        
        # Create new model with updated weights
        self._print("Creating new model with updated weights...")
        config = AutoConfig.from_pretrained(base_model_name)
        
        if model_type == 'causal':
            new_model = AutoModelForCausalLM.from_config(config)
        elif model_type == 'classification':
            new_model = AutoModelForSequenceClassification.from_config(config)
        else:
            new_model = AutoModel.from_config(config)
        
        new_model.load_state_dict(base_state_dict, strict=False)
        
        # Validate the new model if requested
        if validate:
            self._print("Validating new model...")
            if self._validate_model(new_model, base_model_name):
                self._print("Model validation passed!")
            else:
                self._print("Warning: Model validation failed!")
        
        # Save the new model
        self._print(f"Saving updated model to: {output_dir}")
        new_model.save_pretrained(output_dir)
        
        # Save tokenizer
        try:
            tokenizer = AutoTokenizer.from_pretrained(base_model_name)
            tokenizer.save_pretrained(output_dir)
        except Exception as e:
            self._print(f"Warning: Could not save tokenizer: {e}")
        
        # Save replacement log
        replacement_log = {
            'successful_replacements': replacements_made,
            'failed_replacements': replacements_failed,
            'base_model': base_model_name,
            'donor_model': donor_model_name,
            'patterns_used': layers_to_replace
        }
        
        with open(Path(output_dir) / "replacement_log.json", 'w') as f:
            json.dump(replacement_log, f, indent=2)
        
        self._print(f"Replacement complete! {len(replacements_made)} layers replaced.")
        return True
    
    def _validate_model(self, model: torch.nn.Module, reference_model_name: str) -> bool:
        """Basic validation of the manipulated model."""
        try:
            tokenizer = AutoTokenizer.from_pretrained(reference_model_name)
            test_input = "This is a test sentence for model validation."
            
            inputs = tokenizer(test_input, return_tensors="pt", padding=True, truncation=True)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = model(**inputs)
            
            # Check if outputs have expected structure
            if hasattr(outputs, 'last_hidden_state'):
                return not torch.isnan(outputs.last_hidden_state).any()
            elif hasattr(outputs, 'logits'):
                return not torch.isnan(outputs.logits).any()
            else:
                return True  # If we can't check, assume it's okay
            
        except Exception as e:
            self._print(f"Validation error: {e}")
            return False
    
    def merge_models(self, model_names: List[str], weights: List[float], 
                    output_dir: str, model_type: str = 'auto') -> bool:
        """Merge multiple models with specified weights."""
        if len(model_names) != len(weights):
            raise ValueError("Number of models must match number of weights")
        
        if abs(sum(weights) - 1.0) > 1e-6:
            self._print("Warning: Weights don't sum to 1.0, normalizing...")
            weights = [w / sum(weights) for w in weights]
        
        self._print(f"Merging {len(model_names)} models...")
        
        # Load first model as base
        merged_model = self.load_model(model_names[0], model_type)
        merged_state_dict = merged_model.state_dict()
        
        # Zero out the merged state dict and accumulate weighted parameters
        for key in merged_state_dict:
            merged_state_dict[key] = merged_state_dict[key] * weights[0]
        
        # Add weighted parameters from other models
        for i, model_name in enumerate(model_names[1:], 1):
            model = self.load_model(model_name, model_type)
            model_state_dict = model.state_dict()
            
            for key in merged_state_dict:
                if key in model_state_dict:
                    if merged_state_dict[key].shape == model_state_dict[key].shape:
                        merged_state_dict[key] += model_state_dict[key] * weights[i]
                    else:
                        self._print(f"Shape mismatch for {key}, skipping...")
        
        # Create and save merged model
        config = AutoConfig.from_pretrained(model_names[0])
        if model_type == 'causal':
            new_model = AutoModelForCausalLM.from_config(config)
        elif model_type == 'classification':
            new_model = AutoModelForSequenceClassification.from_config(config)
        else:
            new_model = AutoModel.from_config(config)
        
        new_model.load_state_dict(merged_state_dict, strict=False)
        
        # Save merged model
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        new_model.save_pretrained(output_dir)
        
        # Save tokenizer
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_names[0])
            tokenizer.save_pretrained(output_dir)
        except Exception as e:
            self._print(f"Warning: Could not save tokenizer: {e}")
        
        # Save merge log
        merge_log = {
            'models': model_names,
            'weights': weights,
            'output_dir': output_dir
        }
        
        with open(Path(output_dir) / "merge_log.json", 'w') as f:
            json.dump(merge_log, f, indent=2)
        
        self._print("Model merging complete!")
        return True


def main():
    parser = argparse.ArgumentParser(
        description="Advanced Model Layer Manipulation Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Replace a single layer
  python script.py replace --base_model bert-base-uncased --donor_model roberta-base --layers "encoder.layer.11" --output ./output

  # Replace multiple layers with regex pattern
  python script.py replace --base_model bert-base-uncased --donor_model roberta-base --layers "encoder.layer.[0-5]" --output ./output

  # Analyze model compatibility
  python script.py analyze --base_model bert-base-uncased --donor_model roberta-base --output analysis.json

  # Merge two models with equal weights
  python script.py merge --models bert-base-uncased roberta-base --weights 0.5 0.5 --output ./merged_model
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Replace command
    replace_parser = subparsers.add_parser('replace', help='Replace layers between models')
    replace_parser.add_argument('--base_model', type=str, required=True,
                               help='Base model name or path')
    replace_parser.add_argument('--donor_model', type=str, required=True,
                               help='Donor model name or path')
    replace_parser.add_argument('--layers', type=str, nargs='+', required=True,
                               help='Layer identifiers to replace (supports regex)')
    replace_parser.add_argument('--output', type=str, required=True,
                               help='Path to save the updated model')
    replace_parser.add_argument('--model_type', type=str, default='auto',
                               choices=['auto', 'causal', 'classification'],
                               help='Type of model to load')
    replace_parser.add_argument('--no_validate', action='store_true',
                               help='Skip model validation')
    replace_parser.add_argument('--no_backup', action='store_true',
                               help='Skip creating backup of original model')
    
    # Analyze command
    analyze_parser = subparsers.add_parser('analyze', help='Analyze model compatibility')
    analyze_parser.add_argument('--base_model', type=str, required=True,
                               help='Base model name or path')
    analyze_parser.add_argument('--donor_model', type=str, required=True,
                               help='Donor model name or path')
    analyze_parser.add_argument('--output', type=str,
                               help='Path to save analysis results (JSON)')
    
    # Merge command
    merge_parser = subparsers.add_parser('merge', help='Merge multiple models')
    merge_parser.add_argument('--models', type=str, nargs='+', required=True,
                             help='Model names or paths to merge')
    merge_parser.add_argument('--weights', type=float, nargs='+', required=True,
                             help='Weights for each model (must sum to 1.0)')
    merge_parser.add_argument('--output', type=str, required=True,
                             help='Path to save the merged model')
    merge_parser.add_argument('--model_type', type=str, default='auto',
                             choices=['auto', 'causal', 'classification'],
                             help='Type of model to load')
    
    # Global options
    parser.add_argument('--quiet', action='store_true',
                       help='Suppress verbose output')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    manipulator = ModelManipulator(verbose=not args.quiet)
    
    try:
        if args.command == 'replace':
            success = manipulator.replace_layers(
                base_model_name=args.base_model,
                donor_model_name=args.donor_model,
                layers_to_replace=args.layers,
                output_dir=args.output,
                model_type=args.model_type,
                validate=not args.no_validate,
                backup=not args.no_backup
            )
            sys.exit(0 if success else 1)
            
        elif args.command == 'analyze':
            manipulator.analyze_models(
                base_model_name=args.base_model,
                donor_model_name=args.donor_model,
                output_file=args.output
            )
            
        elif args.command == 'merge':
            success = manipulator.merge_models(
                model_names=args.models,
                weights=args.weights,
                output_dir=args.output,
                model_type=args.model_type
            )
            sys.exit(0 if success else 1)
            
    except KeyboardInterrupt:
        print("\nOperation cancelled by user.")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
