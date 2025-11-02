"""
PPO Module for Dynamic Weight Learning
Uses stable-baselines3 PPO to learn optimal weights for agent combination
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from typing import List, Dict, Tuple, Optional
import logging
import os

logger = logging.getLogger(__name__)


class MultiPhishGuardEnv(gym.Env):
    """
    Custom Gym environment for MultiPhishGuard PPO training
    State: Feature vector from FeatureExtractor
    Action: 3 weights [w_text, w_url, w_meta] that sum to 1
    Reward: +1 if correct prediction, -1 if incorrect
    """
    
    def __init__(self, features: List[np.ndarray], labels: List[int], 
                 agent_probs: List[Dict]):
        """
        Initialize environment
        
        Args:
            features: List of feature vectors
            labels: List of true labels (0=legitimate, 1=phishing)
            agent_probs: List of dicts with agent probabilities
                Each dict has 'text', 'url', 'meta' keys with probabilities
        """
        super().__init__()
        
        self.features = np.array(features)
        self.labels = np.array(labels)
        self.agent_probs = agent_probs
        
        self.num_samples = len(features)
        self.current_idx = 0
        
        # Observation space: feature vector
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, 
            shape=(self.features.shape[1],), 
            dtype=np.float32
        )
        
        # Action space: 3 weights (will be normalized to sum=1)
        self.action_space = spaces.Box(
            low=0.0, high=1.0,
            shape=(3,),
            dtype=np.float32
        )
    
    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        """Reset environment"""
        super().reset(seed=seed)
        self.current_idx = 0
        return self.features[self.current_idx].copy(), {}
    
    def step(self, action: np.ndarray):
        """
        Execute one step
        
        Args:
            action: 3 weights [w_text, w_url, w_meta]
        
        Returns:
            observation, reward, terminated, truncated, info
        """
        # Normalize action to sum to 1
        action_normalized = action / (action.sum() + 1e-8)
        action_normalized = np.clip(action_normalized, 0.0, 1.0)
        
        # Get current email data
        current_features = self.features[self.current_idx]
        current_label = self.labels[self.current_idx]
        current_probs = self.agent_probs[self.current_idx]
        
        # Calculate final score using weighted combination
        final_score = (
            action_normalized[0] * current_probs['text'] +
            action_normalized[1] * current_probs['url'] +
            action_normalized[2] * current_probs['meta']
        )
        
        # Make prediction (threshold = 0.5)
        prediction = 1 if final_score >= 0.5 else 0
        
        # Calculate reward
        if prediction == current_label:
            reward = 1.0
        else:
            reward = -1.0
        
        # Move to next sample
        self.current_idx += 1
        terminated = False
        truncated = self.current_idx >= self.num_samples
        
        # Get next observation
        if truncated:
            obs = self.features[0].copy()  # Reset to first sample
        else:
            obs = self.features[self.current_idx].copy()
        
        info = {
            'final_score': float(final_score),
            'prediction': prediction,
            'true_label': int(current_label),
            'weights': action_normalized.tolist()
        }
        
        return obs, reward, terminated, truncated, info


class PPOModule:
    """PPO Module wrapper for training and inference"""
    
    def __init__(self, feature_dim: int, config: Dict):
        """
        Initialize PPO module
        
        Args:
            feature_dim: Dimension of feature vector
            config: Configuration dictionary with PPO hyperparameters
        """
        self.feature_dim = feature_dim
        self.config = config
        
        # Create environment (will be set during training)
        self.env = None
        self.model = None
    
    def create_environment(self, features: List[np.ndarray], labels: List[int],
                          agent_probs: List[Dict]) -> MultiPhishGuardEnv:
        """Create training environment"""
        self.env = MultiPhishGuardEnv(features, labels, agent_probs)
        return self.env
    
    def train(self, features: List[np.ndarray], labels: List[int],
              agent_probs: List[Dict], total_timesteps: int = 100000,
              save_path: str = None, log_dir: str = None):
        """
        Train PPO model
        
        Args:
            features: Feature vectors
            labels: True labels
            agent_probs: Agent probabilities
            total_timesteps: Total training timesteps
            save_path: Path to save model
            log_dir: Directory for logs
        """
        # Create environment
        env = self.create_environment(features, labels, agent_probs)
        
        # Create PPO model
        self.model = PPO(
            "MlpPolicy",
            env,
            learning_rate=self.config.get('learning_rate', 3e-4),
            n_steps=self.config.get('n_steps', 2048),
            batch_size=self.config.get('batch_size', 64),
            n_epochs=self.config.get('n_epochs', 10),
            gamma=self.config.get('gamma', 0.99),
            gae_lambda=self.config.get('gae_lambda', 0.95),
            clip_range=self.config.get('clip_range', 0.2),
            ent_coef=self.config.get('ent_coef', 0.01),
            vf_coef=self.config.get('vf_coef', 0.5),
            max_grad_norm=self.config.get('max_grad_norm', 0.5),
            verbose=1,
            tensorboard_log=log_dir
        )
        
        # Setup checkpoint callback
        callbacks = []
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            checkpoint_callback = CheckpointCallback(
                save_freq=self.config.get('save_freq', 10000),
                save_path=os.path.dirname(save_path),
                name_prefix='ppo_model'
            )
            callbacks.append(checkpoint_callback)
        
        # Train
        logger.info(f"Starting PPO training for {total_timesteps} timesteps...")
        self.model.learn(
            total_timesteps=total_timesteps,
            callback=callbacks if callbacks else None,
            progress_bar=True
        )
        
        # Save final model
        if save_path:
            self.model.save(save_path)
            logger.info(f"Model saved to {save_path}")
    
    def predict_weights(self, observation: np.ndarray) -> np.ndarray:
        """
        Predict weights for given observation
        
        Args:
            observation: Feature vector
        
        Returns:
            Normalized weights [w_text, w_url, w_meta]
        """
        if self.model is None:
            raise ValueError("Model not trained or loaded")
        
        action, _ = self.model.predict(observation, deterministic=True)
        
        # Normalize to sum to 1
        action_normalized = action / (action.sum() + 1e-8)
        action_normalized = np.clip(action_normalized, 0.0, 1.0)
        
        return action_normalized
    
    def load_model(self, model_path: str, env: Optional[MultiPhishGuardEnv] = None):
        """
        Load trained model
        
        Args:
            model_path: Path to saved model
            env: Optional environment (needed for some models)
        """
        if env:
            self.env = env
            self.model = PPO.load(model_path, env=env)
        else:
            self.model = PPO.load(model_path)
        logger.info(f"Model loaded from {model_path}")





