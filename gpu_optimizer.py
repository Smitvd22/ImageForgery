#!/usr/bin/env python3
"""
🚀 GPU Optimizer - Maximum GPU Utilization for Image Forgery Detection
This script will try every possible method to enable GPU computation
"""

import os
import sys
import subprocess
import importlib
import warnings
import torch
import platform

warnings.filterwarnings('ignore')

class GPUOptimizer:
    def __init__(self):
        self.gpu_available = False
        self.cuda_version = None
        self.torch_version = None
        self.device = None
        
    def check_system_gpu(self):
        """Check if NVIDIA GPU is available on system level"""
        print("🔍 Checking system GPU availability...")
        
        try:
            # Check nvidia-smi
            result = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
            if result.returncode == 0:
                print("✅ NVIDIA GPU detected via nvidia-smi")
                # Extract CUDA version
                for line in result.stdout.split('\n'):
                    if 'CUDA Version:' in line:
                        self.cuda_version = line.split('CUDA Version:')[1].strip().split()[0]
                        print(f"✅ CUDA Version: {self.cuda_version}")
                return True
            else:
                print("❌ nvidia-smi not available")
                return False
        except Exception as e:
            print(f"❌ Error checking nvidia-smi: {e}")
            return False
    
    def check_torch_cuda(self):
        """Check current PyTorch CUDA availability"""
        print("🔍 Checking PyTorch CUDA support...")
        
        try:
            import torch
            self.torch_version = torch.__version__
            print(f"📦 PyTorch version: {self.torch_version}")
            
            cuda_available = torch.cuda.is_available()
            print(f"🔥 CUDA available: {cuda_available}")
            
            if cuda_available:
                device_count = torch.cuda.device_count()
                print(f"🎮 GPU devices found: {device_count}")
                
                for i in range(device_count):
                    gpu_name = torch.cuda.get_device_name(i)
                    gpu_memory = torch.cuda.get_device_properties(i).total_memory / 1e9
                    print(f"  GPU {i}: {gpu_name} ({gpu_memory:.1f} GB)")
                
                self.device = torch.device("cuda")
                self.gpu_available = True
                return True
            else:
                print("❌ PyTorch CUDA not available")
                return False
                
        except Exception as e:
            print(f"❌ Error checking PyTorch: {e}")
            return False
    
    def install_cuda_pytorch(self):
        """Install CUDA-enabled PyTorch"""
        print("🚀 Installing CUDA-enabled PyTorch...")
        
        try:
            # Determine the correct PyTorch installation command based on CUDA version
            if self.cuda_version:
                cuda_major = self.cuda_version.split('.')[0]
                cuda_minor = self.cuda_version.split('.')[1]
                
                if cuda_major == '12':
                    if int(cuda_minor) >= 1:
                        torch_url = "https://download.pytorch.org/whl/cu121"
                        print(f"🎯 Installing PyTorch for CUDA 12.1+")
                    else:
                        torch_url = "https://download.pytorch.org/whl/cu118"
                        print(f"🎯 Installing PyTorch for CUDA 11.8")
                elif cuda_major == '11':
                    torch_url = "https://download.pytorch.org/whl/cu118"
                    print(f"🎯 Installing PyTorch for CUDA 11.8")
                else:
                    torch_url = "https://download.pytorch.org/whl/cu118"
                    print(f"🎯 Installing PyTorch for CUDA 11.8 (fallback)")
            else:
                torch_url = "https://download.pytorch.org/whl/cu121"
                print(f"🎯 Installing PyTorch for CUDA 12.1+ (default)")
            
            # Install PyTorch with CUDA
            install_commands = [
                f"pip install torch torchvision torchaudio --index-url {torch_url}",
                "pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121",
                "pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118",
                "pip install torch torchvision torchaudio",  # CPU fallback
            ]
            
            for i, cmd in enumerate(install_commands):
                print(f"🔄 Attempt {i+1}: {cmd}")
                
                try:
                    result = subprocess.run(cmd.split(), capture_output=True, text=True, timeout=300)
                    if result.returncode == 0:
                        print(f"✅ PyTorch installation successful (attempt {i+1})")
                        
                        # Reload torch module
                        importlib.reload(torch)
                        
                        # Check if CUDA is now available
                        if torch.cuda.is_available():
                            print("🎉 CUDA is now available!")
                            self.gpu_available = True
                            self.device = torch.device("cuda")
                            return True
                        else:
                            print(f"⚠️ Installation succeeded but CUDA still not available")
                            if i < len(install_commands) - 1:
                                continue
                    else:
                        print(f"❌ Installation failed: {result.stderr}")
                        if i < len(install_commands) - 1:
                            continue
                            
                except subprocess.TimeoutExpired:
                    print(f"⏰ Installation timeout (attempt {i+1})")
                    continue
                except Exception as e:
                    print(f"❌ Installation error: {e}")
                    continue
            
            print("❌ All PyTorch installation attempts failed")
            return False
            
        except Exception as e:
            print(f"❌ Error during PyTorch installation: {e}")
            return False
    
    def optimize_gpu_settings(self):
        """Optimize GPU settings for maximum performance"""
        print("⚡ Optimizing GPU settings...")
        
        try:
            if self.gpu_available and torch.cuda.is_available():
                # Enable optimizations
                torch.backends.cudnn.benchmark = True
                torch.backends.cudnn.deterministic = False
                torch.backends.cudnn.enabled = True
                
                # Set memory management
                if hasattr(torch.cuda, 'empty_cache'):
                    torch.cuda.empty_cache()
                
                # Get GPU info
                gpu_name = torch.cuda.get_device_name(0)
                gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
                
                print(f"✅ GPU optimizations enabled")
                print(f"🎮 Using: {gpu_name}")
                print(f"💾 Memory: {gpu_memory:.1f} GB")
                
                # Set environment variables for optimal performance
                os.environ['CUDA_LAUNCH_BLOCKING'] = '0'  # Async execution
                os.environ['TORCH_USE_CUDA_DSA'] = '1'    # Use device side assertions
                
                return True
            else:
                print("❌ GPU not available for optimization")
                return False
                
        except Exception as e:
            print(f"❌ Error optimizing GPU: {e}")
            return False
    
    def create_gpu_config(self):
        """Create optimized GPU configuration"""
        print("📝 Creating GPU-optimized configuration...")
        
        gpu_config = f"""
# GPU-Optimized Configuration Generated by GPU Optimizer
import torch

# GPU Detection and Setup
GPU_AVAILABLE = {self.gpu_available}
DEVICE = torch.device("{'cuda' if self.gpu_available else 'cpu'}")
CUDA_VERSION = "{self.cuda_version or 'N/A'}"
TORCH_VERSION = "{self.torch_version or 'N/A'}"

# GPU-Optimized Settings
if GPU_AVAILABLE:
    BATCH_SIZE = 32  # Larger batch size for GPU
    NUM_WORKERS = 8  # More workers for GPU
    PIN_MEMORY = True
    MIXED_PRECISION = True
    GPU_MEMORY_FRACTION = 0.9
    
    # Enable GPU optimizations
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.enabled = True
    
    print(f"🚀 GPU Mode: Using {{torch.cuda.get_device_name(0)}}")
    print(f"💾 GPU Memory: {{torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}} GB")
else:
    BATCH_SIZE = 8   # Conservative for CPU
    NUM_WORKERS = 4  # Fewer workers for CPU
    PIN_MEMORY = False
    MIXED_PRECISION = False
    
    print("🔄 CPU Mode: GPU not available, using CPU fallback")

print(f"🎯 Device: {{DEVICE}}")
print(f"📦 PyTorch: {{TORCH_VERSION}}")
print(f"🔥 CUDA: {{CUDA_VERSION}}")
"""
        
        with open('gpu_config.py', 'w') as f:
            f.write(gpu_config)
        
        print("✅ GPU configuration saved to gpu_config.py")
    
    def test_gpu_computation(self):
        """Test GPU computation with a simple tensor operation"""
        print("🧪 Testing GPU computation...")
        
        try:
            if self.gpu_available:
                # Test tensor operations on GPU
                x = torch.randn(1000, 1000).to(self.device)
                y = torch.randn(1000, 1000).to(self.device)
                
                import time
                start_time = time.time()
                z = torch.mm(x, y)
                torch.cuda.synchronize()  # Wait for completion
                gpu_time = time.time() - start_time
                
                print(f"✅ GPU computation test successful")
                print(f"⚡ GPU matrix multiplication (1000x1000): {gpu_time:.4f}s")
                
                # Test CPU for comparison
                x_cpu = x.cpu()
                y_cpu = y.cpu()
                start_time = time.time()
                z_cpu = torch.mm(x_cpu, y_cpu)
                cpu_time = time.time() - start_time
                
                print(f"🐌 CPU matrix multiplication (1000x1000): {cpu_time:.4f}s")
                print(f"🚀 GPU speedup: {cpu_time/gpu_time:.2f}x")
                
                return True
            else:
                print("❌ Cannot test GPU - not available")
                return False
                
        except Exception as e:
            print(f"❌ GPU computation test failed: {e}")
            return False
    
    def run_optimization(self):
        """Run complete GPU optimization process"""
        print("=" * 60)
        print("🚀 GPU OPTIMIZER - MAXIMUM GPU UTILIZATION")
        print("=" * 60)
        
        # Step 1: Check system GPU
        system_gpu = self.check_system_gpu()
        
        # Step 2: Check PyTorch CUDA
        torch_cuda = self.check_torch_cuda()
        
        # Step 3: If GPU available but PyTorch CUDA not working, install CUDA PyTorch
        if system_gpu and not torch_cuda:
            print("\n🔧 GPU detected but PyTorch CUDA not available. Installing CUDA PyTorch...")
            self.install_cuda_pytorch()
            
            # Recheck after installation
            self.check_torch_cuda()
        
        # Step 4: Optimize GPU settings if available
        if self.gpu_available:
            self.optimize_gpu_settings()
            self.test_gpu_computation()
        else:
            print("\n⚠️ GPU optimization not possible - falling back to CPU")
            self.device = torch.device("cpu")
        
        # Step 5: Create configuration
        self.create_gpu_config()
        
        print("\n" + "=" * 60)
        print("📊 OPTIMIZATION SUMMARY")
        print("=" * 60)
        print(f"🎮 System GPU: {'✅' if system_gpu else '❌'}")
        print(f"🔥 PyTorch CUDA: {'✅' if self.gpu_available else '❌'}")
        print(f"⚡ Device: {self.device}")
        print(f"📦 PyTorch: {self.torch_version}")
        print(f"🚀 CUDA: {self.cuda_version or 'N/A'}")
        
        if self.gpu_available:
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
            print(f"🎯 GPU: {gpu_name} ({gpu_memory:.1f} GB)")
            print("🎉 GPU optimization successful!")
        else:
            print("🔄 Using CPU fallback")
        
        print("=" * 60)
        
        return self.gpu_available

if __name__ == "__main__":
    optimizer = GPUOptimizer()
    success = optimizer.run_optimization()
    
    if success:
        print("\n✅ GPU optimization completed! Your system is ready for GPU-accelerated training.")
        print("🚀 Run: python train_gpu.py")
    else:
        print("\n⚠️ GPU optimization failed, but CPU fallback is ready.")
        print("🔄 Run: python train_enhanced.py")
