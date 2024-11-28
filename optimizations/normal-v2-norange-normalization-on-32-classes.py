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

class MalwareDatasetConfig:
    def __init__(self):
        self.num_classes = 32  # 31 malware + 1 benign
        self.image_size = 128
        self.batch_size = 32
        self.class_names = [
            'Adposhel', 'Agent', 'Allaple', 'Alueron.gen!J', 'Amonetize',
            'Androm', 'Autorun', 'Benign', 'BrowseFox', 'C2LOP.gen!g',
            'Dialplatform.B', 'Dinwod', 'Elex', 'Expiro', 'Fakerean',
            'Fasong', 'HackKMS', 'Hlux', 'Injector', 'InstallCore',
            'Lolyda.AA1', 'Lolyda.AA2', 'MultiPlug', 'Neoreklami',
            'Neshta', 'Regrun', 'Sality', 'Snarasite', 'Stantinko',
            'VBA', 'VBKrypt', 'Vilsel'
        ]

def load_and_preprocess_data(data_dir: str, image_size: int, test_split: float = 0.2):
    # Get all image paths and labels
    image_paths = []
    labels = []
    
    for class_idx, class_name in enumerate(MalwareDatasetConfig().class_names):
        class_dir = os.path.join(data_dir, class_name)
        if not os.path.exists(class_dir):
            continue
            
        for img_path in Path(class_dir).glob('*.png'):
            image_paths.append(str(img_path))
            labels.append(class_idx)
    
    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(
        image_paths, labels,
        test_size=test_split,
        stratify=labels,
        random_state=42
    )
    
    # Create tf.data.Dataset
    def load_image(path, label):
        img = tf.io.read_file(path)
        img = tf.image.decode_png(img, channels=3)
        img = tf.image.resize(img, [image_size, image_size])
        img = tf.image.rgb_to_grayscale(img)
        img = tf.cast(img, tf.float32) / 255.0
        return img, label
    
    train_ds = tf.data.Dataset.from_tensor_slices((X_train, y_train))
    train_ds = train_ds.map(
        lambda x, y: tf.py_function(load_image, [x, y], [tf.float32, tf.int32]),
        num_parallel_calls=tf.data.AUTOTUNE
    )
    train_ds = train_ds.cache().shuffle(1000).batch(32).prefetch(tf.data.AUTOTUNE)
    
    test_ds = tf.data.Dataset.from_tensor_slices((X_test, y_test))
    test_ds = test_ds.map(
        lambda x, y: tf.py_function(load_image, [x, y], [tf.float32, tf.int32]),
        num_parallel_calls=tf.data.AUTOTUNE
    )
    test_ds = test_ds.batch(32).prefetch(tf.data.AUTOTUNE)
    
    return train_ds, test_ds

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
        
        # Stronger feature extraction
        x = img
        channels = [64, 128, 256, 512]  # Increased channels
        
        for ch in channels:
            x = tf.keras.layers.Conv2D(ch, 4, strides=2, padding='same')(x)
            x = tf.keras.layers.LeakyReLU(0.2)(x)
            x = tf.keras.layers.Dropout(0.3)(x)  # Add dropout
            
            # Spectral normalization for stability
            x = tf.keras.layers.Conv2D(ch*2, 4, strides=1, padding='same',
                kernel_regularizer=tf.keras.regularizers.l2(1e-4))(x)
            x = tf.keras.layers.LeakyReLU(0.2)(x)
            x = tf.keras.layers.BatchNormalization(momentum=0.8)(x)
        
        x = tf.keras.layers.GlobalAveragePooling2D()(x)
        x = tf.keras.layers.Concatenate()([x, label])
        
        # Deeper classification head
        x = tf.keras.layers.Dense(512)(x)
        x = tf.keras.layers.LeakyReLU(0.2)(x)
        x = tf.keras.layers.Dropout(0.3)(x)
        x = tf.keras.layers.Dense(256)(x)
        x = tf.keras.layers.LeakyReLU(0.2)(x)
        x = tf.keras.layers.Dense(self.config.num_variants)(x)
        
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
            # Generate variants with more diversity
            z_base = tf.random.normal((batch_size, self.config.latent_dim))
            w_base = self.mapping_network([z_base, labels])
            
            variants_list = []
            for i in range(self.config.num_variants):
                z_new = tf.random.normal((batch_size, self.config.latent_dim))
                w_new = self.mapping_network([z_new, labels])
                # Progressive mixing ratio
                alpha = 0.5 + (i / self.config.num_variants) * 0.3
                w_mixed = w_base * alpha + w_new * (1.0 - alpha)
                variant = self.generator(w_mixed)
                variants_list.append(variant)
            
            variants_stacked = tf.stack(variants_list)
            
            # Compute discriminator outputs
            variant_logits = self.discriminator([
                tf.reshape(variants_stacked, 
                          [-1, self.config.image_size, self.config.image_size, 
                           self.config.num_channels]),
                tf.tile(labels, [self.config.num_variants, 1])
            ])
            
            # Wasserstein loss with gradient penalty
            gradient_penalty = self._gradient_penalty(real_images, variants_stacked[0])
            
            variant_labels = tf.cast(tf.eye(self.config.num_variants), dtype=tf.float32)
            variant_labels = tf.tile(variant_labels, [batch_size, 1])
            
            # Modified loss calculation
            variant_loss = tf.reduce_mean(
                tf.keras.losses.categorical_crossentropy(
                    variant_labels, 
                    variant_logits,
                    from_logits=True
                )
            ) + 10.0 * gradient_penalty
            
            diversity_loss = -tf.reduce_mean(
                tf.abs(variants_stacked[:, None] - variants_stacked[None, :])
            ) * self.config.diversity_weight
            
            generator_loss = -variant_loss + diversity_loss

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
        if (epoch + 1) % 20 != 0:
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

    def _gradient_penalty(self, real_images, generated_images):
        """Compute gradient penalty for WGAN-GP"""
        batch_size = tf.shape(real_images)[0]
        alpha = tf.random.uniform([batch_size, 1, 1, 1], 0.0, 1.0)
        diff = generated_images - real_images
        interpolated = real_images + alpha * diff

        with tf.GradientTape() as gp_tape:
            gp_tape.watch(interpolated)
            # Get discriminator output
            pred = self.discriminator([interpolated, tf.zeros((batch_size, self.num_classes))])

        grads = gp_tape.gradient(pred, interpolated)
        norm = tf.sqrt(tf.reduce_sum(tf.square(grads), axis=[1, 2, 3]))
        gp = tf.reduce_mean((norm - 1.0) ** 2)
        
        return gp

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

        epoch_losses = []
        for batch_idx, (images, labels) in enumerate(train_ds):
            try:
                losses = model.train_step(images, labels)
                epoch_losses.append(losses)

                if batch_idx % 10 == 0:
                    print(f"Batch {batch_idx}: variant_loss={losses['variant_loss']:.4f}, "
                          f"generator_loss={losses['generator_loss']:.4f}")

            except tf.errors.ResourceExhaustedError:
                print("GPU memory exhausted, skipping batch")
                tf.keras.backend.clear_session()
                continue

        # Calculate epoch averages
        avg_losses = {k: float(np.mean([x[k] for x in epoch_losses]))
                     for k in epoch_losses[0].keys()}
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
