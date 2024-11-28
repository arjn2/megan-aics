import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
from pathlib import Path
from dataclasses import dataclass
import json
from typing import Dict , Tuple
import logging
import absl.logging
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

# Initialize ABSL logging
absl.logging.set_verbosity(absl.logging.ERROR)
logging.root.removeHandler(absl.logging._absl_handler)
absl.logging._warn_preinit_stderr = False

# Set random seeds
import random
random.seed(42)
tf.random.set_seed(42)
np.random.seed(42)

# Configure GPU
physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    for device in physical_devices:
        tf.config.experimental.set_memory_growth(device, True)
    print(f"GPU(s) detected: {len(physical_devices)}")
else:
    print("No GPU detected, using CPU")

@dataclass
class StyleGANConfig:
    image_size: int = 128
    latent_dim: int = 64
    mapping_layers: int = 2
    style_dim: int = 64
    num_channels: int = 1
    batch_size: int = 8    # Increased slightly
    learning_rate: float = 0.0001
    epochs: int = 50  # Reduced from 100
    checkpoint_dir: str = './checkpoints'
    gradient_clip: float = 1.0
    use_mixed_precision: bool = False  # Disabled mixed precision
    num_variants: int = 4      # Number of variants per malware
    style_mixing_prob: float = 0.9  # Probability of style mixing
    diversity_weight: float = 0.1    # Weight for variant diversity loss
    early_stopping_patience: int = 5  # Add early stopping
    loss_plateau_threshold: float = 1e-4
    plateau_patience: int = 10
    min_batches_per_epoch: int = 50

# Remove mixed precision policy
# tf.keras.mixed_precision.set_global_policy('mixed_float16')

config = StyleGANConfig()

def preprocess_dataset(data_dir: str, image_size: int, batch_size: int):
    try:
        class_names = sorted([d.name for d in Path(data_dir).iterdir() if d.is_dir()])
        print(f"Found classes: {class_names}")

        # Optimize dataset loading
        dataset = tf.keras.utils.image_dataset_from_directory(
            data_dir,
            labels='inferred',
            label_mode='categorical',
            class_names=class_names,
            color_mode='rgb',
            batch_size=batch_size,
            image_size=(image_size, image_size),
            shuffle=True,
            seed=42
        )

        # Memory efficient preprocessing
        options = tf.data.Options()
        options.experimental_optimization.map_parallelization = True
        options.experimental_optimization.parallel_batch = True
        options.threading.private_threadpool_size = 4  # Add threading
        options.threading.max_intra_op_parallelism = 2

        def optimize_image(image, label):
            image = tf.cast(image, tf.float32)
            image = tf.image.rgb_to_grayscale(image)
            image = (image / 127.5) - 1.0
            return image, label

        preprocessed_ds = (dataset
            .with_options(options)
            .map(optimize_image, num_parallel_calls=tf.data.AUTOTUNE)
            .cache()
            .prefetch(tf.data.AUTOTUNE))

        return preprocessed_ds, class_names

    except Exception as e:
        print(f"Error in dataset preprocessing: {str(e)}")
        raise

# Add progress tracking
def add_progress_callback(dataset, total_files):
    counter = 0
    def progress_callback(*args):
        nonlocal counter
        counter += 1
        if counter % 100 == 0:
            print(f"Processed {counter}/{total_files} files...")

    return dataset.map(
        lambda x, y: (x, y),
        num_parallel_calls=tf.data.AUTOTUNE
    ).apply(tf.data.experimental.snapshot(
        path="./dataset_cache",
        compression="AUTO",
        progress_callback=progress_callback
    ))

def load_and_preprocess_data(data_dir: str, image_size: int, test_split: float = 0.2):
    """Load, normalize and split dataset"""
    
    try:
        # Load all images and labels
        dataset = tf.keras.utils.image_dataset_from_directory(
            data_dir,
            labels='inferred',
            label_mode='categorical',
            color_mode='rgb',
            batch_size=None,  # Load all images
            image_size=(image_size, image_size),
            shuffle=True,
            seed=42
        )

        # Convert to numpy for splitting
        images = []
        labels = []
        for image, label in dataset:
            images.append(image)
            labels.append(label)
            
        images = np.array(images)
        labels = np.array(labels)

        # Normalize images to [-1, 1]
        images = (images.astype('float32') - 127.5) / 127.5

        # Split into train/test
        X_train, X_test, y_train, y_test = train_test_split(
            images, labels,
            test_size=test_split,
            random_state=42,
            stratify=labels
        )

        print(f"Training set: {len(X_train)} images")
        print(f"Test set: {len(X_test)} images")
        
        # Convert back to tf.data.Dataset
        train_ds = tf.data.Dataset.from_tensor_slices((X_train, y_train))
        test_ds = tf.data.Dataset.from_tensor_slices((X_test, y_test))
        
        # Add batching and prefetching
        train_ds = train_ds.shuffle(1000).batch(32).prefetch(tf.data.AUTOTUNE)
        test_ds = test_ds.batch(32).prefetch(tf.data.AUTOTUNE)

        return train_ds, test_ds

    except Exception as e:
        print(f"Error in data preprocessing: {str(e)}")
        raise

def normalize_data(data_dir: str, image_size: int, batch_size: int):
    """Normalize dataset with proper validation and tracking"""
    try:
        # Track statistics
        stats = {
            'min_val': float('inf'),
            'max_val': float('-inf'),
            'mean': 0,
            'std': 0
        }
        
        # Load dataset
        dataset = tf.keras.utils.image_dataset_from_directory(
            data_dir,
            labels='inferred',
            label_mode='categorical',
            color_mode='rgb',
            batch_size=batch_size,
            image_size=(image_size, image_size),
            shuffle=True,
            seed=42
        )
        
        # Separate normalization from augmentation
        normalization_pipeline = tf.keras.Sequential([
            tf.keras.layers.Rescaling(1./127.5, offset=-1),  # Scale to [-1, 1]
        ], name='normalization')
        
        augmentation_pipeline = tf.keras.Sequential([
            tf.keras.layers.RandomFlip("horizontal"),
            tf.keras.layers.RandomRotation(0.2),
        ], name='augmentation')
        
        # Track statistics for validation
        def update_stats(image, label):
            image = tf.cast(image, tf.float32)
            stats['min_val'] = tf.minimum(stats['min_val'], tf.reduce_min(image))
            stats['max_val'] = tf.maximum(stats['max_val'], tf.reduce_max(image))
            stats['mean'] = tf.reduce_mean(image)
            stats['std'] = tf.math.reduce_std(image)
            return image, label
        
        # Apply preprocessing
        normalized_ds = dataset.map(
            update_stats,
            num_parallel_calls=tf.data.AUTOTUNE
        ).map(
            lambda x, y: (normalization_pipeline(x), y),
            num_parallel_calls=tf.data.AUTOTUNE
        )
        
        # Add augmentation only for training
        train_ds = normalized_ds.map(
            lambda x, y: (augmentation_pipeline(x), y),
            num_parallel_calls=tf.data.AUTOTUNE
        ).cache().prefetch(tf.data.AUTOTUNE)
        
        # Validation pipeline without augmentation
        val_ds = normalized_ds.cache().prefetch(tf.data.AUTOTUNE)
        
        return train_ds, val_ds, stats

    except Exception as e:
        print(f"Error in data normalization: {str(e)}")
        raise

class MalwareStyleGAN:
    def __init__(self, config: StyleGANConfig, num_classes: int):
        self.config = config
        self.num_classes = num_classes

        # Remove encoder, keep other components
        self.mapping_network = self.build_mapping_network()
        self.generator = self.build_memory_efficient_generator()
        self.discriminator = self.build_memory_efficient_discriminator()

        # Remove encoder optimizer
        self.d_optimizer = tf.keras.optimizers.Adam(config.learning_rate)
        self.g_optimizer = tf.keras.optimizers.Adam(config.learning_rate)
        self.m_optimizer = tf.keras.optimizers.Adam(config.learning_rate)

    def _initialize_optimizers(self, dummy_input, dummy_label):
        """Initialize optimizers with their respective variables"""
        with tf.GradientTape(persistent=True) as tape:
            w = self.mapping_network([dummy_input, dummy_label])
            fake = self.generator(w)
            _ = self.discriminator([fake, dummy_label])

        # Apply dummy gradients to initialize optimizer variables
        self.d_optimizer.apply_gradients(
            zip([tf.zeros_like(v) for v in self.discriminator.trainable_variables],
                self.discriminator.trainable_variables))
        self.g_optimizer.apply_gradients(
            zip([tf.zeros_like(v) for v in self.generator.trainable_variables],
                self.generator.trainable_variables))
        self.m_optimizer.apply_gradients(
            zip([tf.zeros_like(v) for v in self.mapping_network.trainable_variables],
                self.mapping_network.trainable_variables))

    def build_mapping_network(self):
        z = tf.keras.layers.Input(shape=(self.config.latent_dim,))
        y = tf.keras.layers.Input(shape=(self.num_classes,))

        x = tf.keras.layers.Concatenate()([z, y])

        for _ in range(self.config.mapping_layers):
            x = tf.keras.layers.Dense(self.config.style_dim)(x)
            x = tf.keras.layers.LeakyReLU(0.2)(x)

        return tf.keras.Model([z, y], x, name='mapping_network')

    def build_memory_efficient_generator(self):
        style = tf.keras.layers.Input(shape=(self.config.style_dim,))

        # Add pattern-preserving layers
        x = tf.keras.layers.Dense(8 * 8 * 256)(style)
        x = tf.keras.layers.Reshape((8, 8, 256))(x)

        # Add residual connections to maintain patterns
        channels = [256, 128, 64, 32]
        for ch in channels:
            skip = x
            x = tf.keras.layers.UpSampling2D()(x)
            x = tf.keras.layers.Conv2D(ch, 3, padding='same')(x)
            x = tf.keras.layers.LeakyReLU(0.2)(x)
            x = tf.keras.layers.BatchNormalization(momentum=0.8)(x)
            if x.shape[1:3] == skip.shape[1:3]:
                x = tf.keras.layers.Add()([x, skip])

        x = tf.keras.layers.Conv2D(self.config.num_channels, 1,
                                  activation='tanh')(x)

        return tf.keras.Model(style, x, name='generator')

    def build_memory_efficient_discriminator(self):
        img = tf.keras.layers.Input(shape=(self.config.image_size,
                                          self.config.image_size,
                                          self.config.num_channels))
        label = tf.keras.layers.Input(shape=(self.num_classes,))

        x = img
        channels = [32, 64, 128, 256]

        # Add residual connections
        for ch in channels:
            residual = x
            x = tf.keras.layers.Conv2D(ch, 3, strides=2, padding='same')(x)
            x = tf.keras.layers.LeakyReLU(0.2)(x)
            x = tf.keras.layers.BatchNormalization(momentum=0.8)(x)
            if x.shape[1:3] == residual.shape[1:3]:
                x = tf.keras.layers.Add()([x, residual])

        x = tf.keras.layers.GlobalAveragePooling2D()(x)
        x = tf.keras.layers.Concatenate()([x, label])
        x = tf.keras.layers.Dense(256)(x)
        x = tf.keras.layers.LeakyReLU(0.2)(x)
        x = tf.keras.layers.Dense(self.config.num_variants, activation='sigmoid')(x)

        return tf.keras.Model([img, label], x, name='discriminator')

    def generate_variants(self, z, labels, num_variants):
        """Generate diverse malware variants using style mixing"""
        variants = []
        latent_codes = []

        # Generate base style
        w_base = self.mapping_network([z, labels])

        for _ in range(num_variants):
            if tf.random.uniform([]) < self.config.style_mixing_prob:
                # Generate new style for mixing
                z_new = tf.random.normal((tf.shape(z)))
                w_new = self.mapping_network([z_new, labels])

                # Mix styles
                mixing_weights = tf.random.uniform([tf.shape(w_base)[1]], 0, 1)
                w_mixed = w_base * mixing_weights + w_new * (1 - mixing_weights)
                latent_codes.append(w_mixed)
            else:
                # Use original style with small perturbation
                w_perturbed = w_base + tf.random.normal(tf.shape(w_base)) * 0.1
                latent_codes.append(w_perturbed)

            variant = self.generator(latent_codes[-1])
            variants.append(variant)

        return variants, latent_codes

    @tf.function
    def train_step(self, real_images, labels):
        batch_size = tf.shape(real_images)[0]

        with tf.GradientTape(persistent=True) as tape:
            z_base = tf.random.normal((batch_size, self.config.latent_dim))
            w_base = self.mapping_network([z_base, labels])

            variants_list = []
            for _ in range(self.config.num_variants):
                z_new = tf.random.normal((batch_size, self.config.latent_dim))
                w_new = self.mapping_network([z_new, labels])
                w_mixed = w_base * 0.7 + w_new * 0.3
                variant = self.generator(w_mixed)
                variants_list.append(variant)

            variants_stacked = tf.stack(variants_list)

            variant_logits = self.discriminator([
                tf.reshape(variants_stacked,
                          [-1, self.config.image_size, self.config.image_size,
                           self.config.num_channels]),
                tf.tile(labels, [self.config.num_variants, 1])
            ])

            variant_labels = tf.cast(tf.eye(self.config.num_variants), dtype=tf.float32)
            variant_labels = tf.tile(variant_labels, [batch_size, 1])

            variant_loss = tf.clip_by_value(
                tf.reduce_mean(
                    tf.nn.sigmoid_cross_entropy_with_logits(
                        labels=variant_labels,
                        logits=variant_logits
                    )
                ), -1.0, 1.0)

            diversity_loss = tf.clip_by_value(
                -tf.reduce_mean(
                    tf.abs(variants_stacked[:, None] - variants_stacked[None, :])
                ) * 0.1, -1.0, 1.0)

            generator_loss = tf.clip_by_value(-variant_loss + 
                                            self.config.diversity_weight * diversity_loss,
                                            -1.0, 1.0)

        # Update gradients without encoder
        gradients = [
            (variant_loss, self.discriminator.trainable_variables, self.d_optimizer),
            (generator_loss, self.generator.trainable_variables, self.g_optimizer),
            (generator_loss, self.mapping_network.trainable_variables, self.m_optimizer)
        ]

        for loss, vars, optimizer in gradients:
            grads = tape.gradient(loss, vars)
            if grads is not None:
                clipped_grads, _ = tf.clip_by_global_norm(grads, self.config.gradient_clip)
                optimizer.apply_gradients(zip(clipped_grads, vars))

        return {
            'variant_loss': variant_loss,
            'generator_loss': generator_loss,
            'diversity_loss': diversity_loss
        }

    def save_model(self, path: str, class_indices: Dict):
        Path(path).mkdir(parents=True, exist_ok=True)

        # Save models with proper extensions
        self.mapping_network.save(os.path.join(path, 'mapping_network.keras'))
        self.generator.save(os.path.join(path, 'generator.keras'))
        self.discriminator.save(os.path.join(path, 'discriminator.keras'))

        # Save class indices
        with open(os.path.join(path, 'class_indices.json'), 'w') as f:
            json.dump(class_indices, f)

    # Update load function
    def load_model(self, path: str):
        self.mapping_network = tf.keras.models.load_model(
            os.path.join(path, 'mapping_network.keras'))
        self.generator = tf.keras.models.load_model(
            os.path.join(path, 'generator.keras'))
        self.discriminator = tf.keras.models.load_model(
            os.path.join(path, 'discriminator.keras'))

    def evaluate_model(self, val_ds, epoch):
        """Evaluate model with improved metrics handling"""
        if (epoch + 1) % 2 != 0:
            return
            
        predictions = []
        true_labels = []
        
        # Collect predictions
        for images, labels in val_ds:
            z = tf.random.normal((tf.shape(images)[0], self.config.latent_dim))
            w = self.mapping_network([z, labels])
            fake_images = self.generator(w)
            
            pred = self.discriminator([fake_images, labels])
            predictions.append(tf.argmax(pred, axis=1))
            true_labels.append(tf.argmax(labels, axis=1))
        
        # Convert to numpy arrays
        predictions = tf.concat(predictions, axis=0).numpy()
        true_labels = tf.concat(true_labels, axis=0).numpy()
        
        # Calculate metrics with zero division handling
        conf_matrix = confusion_matrix(true_labels, predictions)
        accuracy = tf.reduce_mean(tf.cast(tf.equal(predictions, true_labels), tf.float32))
        
        report = classification_report(
            true_labels, 
            predictions,
            zero_division=1,  # Handle zero division
            digits=4  # Increase precision
        )
        
        # Log metrics
        print(f"\nEvaluation Metrics at Epoch {epoch + 1}:")
        print(f"Accuracy: {accuracy:.4f}")
        print("\nClassification Report:")
        print(report)
        
        # Plot confusion matrix
        plt.figure(figsize=(12, 10))
        sns.heatmap(
            conf_matrix, 
            annot=True, 
            fmt='d',
            cmap='Blues',
            xticklabels=range(self.config.num_variants),
            yticklabels=range(self.config.num_variants)
        )
        plt.title(f'Confusion Matrix - Epoch {epoch + 1}')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.tight_layout()
        plt.savefig(f'metrics/confusion_matrix_epoch_{epoch + 1}.png', dpi=300)
        plt.close()
        
        return {
            'accuracy': float(accuracy),
            'confusion_matrix': conf_matrix.tolist(),
            'classification_report': report
        }

def train_with_early_batch_stopping(model, train_ds, epoch):
    losses_history = []
    plateau_counter = 0
    epoch_losses = []
    
    for batch_idx, (images, labels) in enumerate(train_ds):
        batch_start_time = time.time()
        losses = model.train_step(images, labels)
        batch_time = time.time() - batch_start_time
        
        # Convert tensor to float for history
        variant_loss = float(losses['variant_loss'])
        losses_history.append(variant_loss)
        epoch_losses.append({k: float(v) for k, v in losses.items()})
        
        # Print progress
        if batch_idx % 10 == 0:
            print(f"Batch {batch_idx}: variant_loss={variant_loss:.4f}, "
                  f"generator_loss={float(losses['generator_loss']):.4f}, "
                  f"batch_time={batch_time:.2f}s")
        
        # Check for loss plateau
        if len(losses_history) > 1:
            loss_diff = abs(losses_history[-1] - losses_history[-2])
            if loss_diff < config.loss_plateau_threshold:
                plateau_counter += 1
            else:
                plateau_counter = 0
        
        # Early stopping conditions
        if (plateau_counter >= config.plateau_patience and 
            batch_idx >= config.min_batches_per_epoch):
            print(f"Early batch stopping at batch {batch_idx} due to loss plateau")
            break
            
    return epoch_losses

def main():
    # Configure GPU with simpler strategy
    physical_devices = tf.config.list_physical_devices('GPU')
    if (physical_devices):
        print(f"GPU(s) detected: {len(physical_devices)}")
        # Configure memory growth
        for device in physical_devices:
            tf.config.experimental.set_memory_growth(device, True)
    else:
        print("No GPU detected, using CPU")

    # Update paths
    train_path = os.path.join('sample_data', 'dataset', 'train')
    val_path = os.path.join('sample_data', 'dataset', 'val')

    # Create directories
    os.makedirs(config.checkpoint_dir, exist_ok=True)
    os.makedirs('samples', exist_ok=True)
    os.makedirs('metrics', exist_ok=True)

    # Load datasets without distribution
    train_ds, class_names = preprocess_dataset(train_path,
                                             config.image_size,
                                             config.batch_size)
    val_ds, _ = preprocess_dataset(val_path,
                                 config.image_size,
                                 config.batch_size)

    # Get class indices
    raw_ds = tf.keras.utils.image_dataset_from_directory(
        './sample_data/dataset/train',
        label_mode='categorical'
    )
    class_indices = {name: i for i, name in enumerate(raw_ds.class_names)}

    # Initialize model
    model = MalwareStyleGAN(config, len(class_indices))

    # Training loop
    best_loss = float('inf')
    patience_counter = 0

    for epoch in range(config.epochs):
        print(f"Epoch {epoch+1}/{config.epochs}")

        # Get losses as list of dicts
        epoch_losses = train_with_early_batch_stopping(model, train_ds, epoch)
        
        # Calculate epoch averages safely
        if epoch_losses:
            avg_losses = {
                k: float(np.mean([x[k] for x in epoch_losses]))
                for k in epoch_losses[0].keys()
            }
            print(f"Epoch {epoch+1} Averages:")
            for k, v in avg_losses.items():
                print(f"{k}: {v:.4f}")

        # Save checkpoint after each epoch
        checkpoint_path = os.path.join(config.checkpoint_dir, f'epoch_{epoch+1}')
        model.save_model(checkpoint_path, class_indices)
        print(f"Saved checkpoint at {checkpoint_path}")

        # Generate samples every 10 epochs
        if epoch % 10 == 0:
            z = tf.random.normal((4, config.latent_dim))
            labels = tf.one_hot(range(4), len(class_indices))
            w = model.mapping_network([z, labels])
            fake_images = model.generator(w)

            plt.figure(figsize=(10, 10))
            for i in range(4):
                plt.subplot(2, 2, i+1)
                plt.imshow(fake_images[i] * 0.5 + 0.5, cmap='gray')
                plt.axis('off')
            plt.savefig(f'samples/epoch_{epoch+1}.png')
            plt.close()

        # Check early stopping
        current_loss = avg_losses['variant_loss']
        if current_loss < best_loss:
            best_loss = current_loss
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= config.early_stopping_patience:
            print("Early stopping triggered")
            break

        # Evaluate every 5 epochs
        model.evaluate_model(val_ds, epoch)

        # Save metrics
        metrics = {
            'epoch': epoch + 1,
            'losses': avg_losses,
            'accuracy': float(accuracy) if 'accuracy' in locals() else None
        }
        
        with open(os.path.join('metrics', f'metrics_epoch_{epoch+1}.json'), 'w') as f:
            json.dump(metrics, f, indent=2)

if __name__ == "__main__":
    main()

import absl.logging
absl.logging.set_verbosity(absl.logging.ERROR)
absl.logging.use_absl_handler()

# Example usage:
if __name__ == "__main__":
    data_dir = "sample_data/dataset/train"
    image_size = 128
    
    train_ds, test_ds = load_and_preprocess_data(
        data_dir=data_dir,
        image_size=image_size,
        test_split=0.2
    )
