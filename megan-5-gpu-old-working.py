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

class MalwareStyleGAN:
    def __init__(self, config: StyleGANConfig, num_classes: int):
        self.config = config
        self.num_classes = num_classes
        
        # Initialize models
        self.mapping_network = self.build_mapping_network()
        self.generator = self.build_memory_efficient_generator()
        self.encoder = self.build_encoder()
        self.discriminator = self.build_memory_efficient_discriminator()
        
        # Create separate optimizers
        self.d_optimizer = tf.keras.optimizers.Adam(config.learning_rate)
        self.g_optimizer = tf.keras.optimizers.Adam(config.learning_rate)
        self.m_optimizer = tf.keras.optimizers.Adam(config.learning_rate)
        self.e_optimizer = tf.keras.optimizers.Adam(config.learning_rate)
        
        # Initialize optimizers with their respective variables
        dummy_input = tf.zeros((1, config.latent_dim))
        dummy_label = tf.zeros((1, num_classes))
        self._initialize_optimizers(dummy_input, dummy_label)
    
    def _initialize_optimizers(self, dummy_input, dummy_label):
        """Initialize optimizers with their respective variables"""
        with tf.GradientTape(persistent=True) as tape:
            w = self.mapping_network([dummy_input, dummy_label])
            fake = self.generator(w)
            _ = self.discriminator([fake, dummy_label])
            _ = self.encoder(fake)
            
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
        self.e_optimizer.apply_gradients(
            zip([tf.zeros_like(v) for v in self.encoder.trainable_variables],
                self.encoder.trainable_variables))

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
        
        # Smaller initial size
        x = tf.keras.layers.Dense(8 * 8 * 256)(style)
        x = tf.keras.layers.Reshape((8, 8, 256))(x)
        
        # Reduced channels
        channels = [256, 128, 64, 32]
        for ch in channels:
            x = tf.keras.layers.UpSampling2D()(x)
            x = tf.keras.layers.Conv2D(ch, 3, padding='same', 
                                     kernel_initializer='he_normal')(x)
            x = tf.keras.layers.LeakyReLU(0.2)(x)
            x = tf.keras.layers.BatchNormalization(momentum=0.8)(x)
        
        x = tf.keras.layers.Conv2D(self.config.num_channels, 1, 
                                  activation='tanh')(x)
        
        return tf.keras.Model(style, x, name='generator')
    
    def build_encoder(self):
        img = tf.keras.layers.Input(shape=(None, None, self.config.num_channels))
        
        x = img
        channels = [32, 64, 128, 256, 512]
        
        for ch in channels:
            x = tf.keras.layers.Conv2D(ch, 3, strides=2, padding='same')(x)
            x = tf.keras.layers.LeakyReLU(0.2)(x)
            x = tf.keras.layers.BatchNormalization()(x)
        
        x = tf.keras.layers.GlobalAveragePooling2D()(x)
        x = tf.keras.layers.Dense(self.config.style_dim)(x)
        
        return tf.keras.Model(img, x, name='encoder')
    
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
        # Change output to match number of variants
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
            # Cast inputs to float32
            real_images = tf.cast(real_images, tf.float32)
            labels = tf.cast(labels, tf.float32)
            
            # Generate base variant
            z_base = tf.random.normal((batch_size, self.config.latent_dim), 
                                    dtype=tf.float32,
                                    stddev=1.0)  # Control variance
            w_base = self.mapping_network([z_base, labels])
            
            # Generate variants with controlled noise
            variants_list = []
            w_variants_list = []
            
            # Add base variant
            base_variant = self.generator(w_base)
            variants_list.append(base_variant)
            w_variants_list.append(w_base)
            
            # Generate additional variants
            for _ in range(self.config.num_variants - 1):
                z_new = tf.random.normal((batch_size, self.config.latent_dim), 
                                       dtype=tf.float32,
                                       stddev=0.5)  # Reduced variance
                w_new = self.mapping_network([z_new, labels])
                w_mixed = w_base * 0.7 + w_new * 0.3  # Adjusted mixing ratio
                variant = self.generator(w_mixed)
                variants_list.append(variant)
                w_variants_list.append(w_mixed)
            
            # Stack and normalize variants
            variants_stacked = tf.stack(variants_list)
            variants_stacked = tf.clip_by_value(variants_stacked, -1.0, 1.0)
            
            # Compute discriminator outputs
            variant_logits = self.discriminator([
                tf.reshape(variants_stacked, 
                          [-1, self.config.image_size, self.config.image_size, self.config.num_channels]),
                tf.tile(labels, [self.config.num_variants, 1])
            ])
            
            # Loss calculations with stable numerics
            variant_labels = tf.cast(tf.eye(self.config.num_variants), dtype=tf.float32)
            variant_labels = tf.tile(variant_labels, [batch_size, 1])
            
            # Use stable cross entropy
            variant_loss = tf.reduce_mean(
                tf.nn.softmax_cross_entropy_with_logits(
                    labels=variant_labels,
                    logits=variant_logits
                ))
            
            # Normalized diversity loss
            diversity_loss = -tf.reduce_mean(
                tf.abs(variants_stacked[:, None] - variants_stacked[None, :])
            ) * 0.1  # Scale factor
            
            generator_loss = -variant_loss + self.config.diversity_weight * diversity_loss
            encoder_loss = tf.reduce_mean(tf.square(self.encoder(real_images) - w_base))
        
        # Apply gradients with clipping
        gradients = [
            (variant_loss, self.discriminator.trainable_variables, self.d_optimizer),
            (generator_loss, self.generator.trainable_variables, self.g_optimizer),
            (generator_loss, self.mapping_network.trainable_variables, self.m_optimizer),
            (encoder_loss, self.encoder.trainable_variables, self.e_optimizer)
        ]
        
        for loss, vars, optimizer in gradients:
            grads = tape.gradient(loss, vars)
            if grads is not None:
                clipped_grads, _ = tf.clip_by_global_norm(grads, self.config.gradient_clip)
                optimizer.apply_gradients(zip(clipped_grads, vars))
        
        # Return losses with consistent keys
        return {
            'variant_loss': variant_loss,
            'generator_loss': generator_loss,
            'encoder_loss': encoder_loss,
            'diversity_loss': diversity_loss
        }
    
    def save_model(self, path: str, class_indices: Dict):
        Path(path).mkdir(parents=True, exist_ok=True)
        
        # Save models with proper extensions
        self.mapping_network.save(os.path.join(path, 'mapping_network.keras'))
        self.generator.save(os.path.join(path, 'generator.keras'))
        self.encoder.save(os.path.join(path, 'encoder.keras'))
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
        self.encoder = tf.keras.models.load_model(
            os.path.join(path, 'encoder.keras'))
        self.discriminator = tf.keras.models.load_model(
            os.path.join(path, 'discriminator.keras'))

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
                          f"generator_loss={losses['generator_loss']:.4f}, "
                          f"encoder_loss={losses['encoder_loss']:.4f}")
            
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

if __name__ == "__main__":
    main()

import absl.logging
absl.logging.set_verbosity(absl.logging.ERROR)
absl.logging.use_absl_handler()
