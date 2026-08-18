# Rust/WASM SIMD Vector Tensor Compute Engine ⚡ 🏎️

<div align="center">

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge)](https://opensource.org/licenses/MIT)
[![TypeScript](https://img.shields.io/badge/TypeScript-5.3+-3178C6?style=for-the-badge&logo=typescript&logoColor=white)](https://www.typescriptlang.org/)
[![Version](https://img.shields.io/badge/Version-v5.0.0%20Ultra-00d2ff?style=for-the-badge)](https://github.com/anderdebona/vector-tensor-wasm-engine)
[![CI/CD](https://img.shields.io/badge/CI%2FCD-Passing%20100%25-success?style=for-the-badge&logo=githubactions)](https://github.com/anderdebona/vector-tensor-wasm-engine/actions)

<br />

**PhD-Grade Vector Tensor Compute Engine: Inverted File Product Quantization (IVF-PQ), Compressed Sparse Row (CSR) SpMM, 128-bit SIMD Vectorization & KMeans Voronoi Partitions**

*Engineered with precision by **[anderdebona](https://github.com/anderdebona)***

</div>

---

## 📌 Technical Summary & Scientific Background

This repository delivers a **hardware-accelerated Vector & Tensor compute engine** targeting WebAssembly with 128-bit SIMD intrinsics. It implements Inverted File Product Quantization (IVF-PQ) with Asymmetric Distance Computation (ADC), Compressed Sparse Row (CSR) sparse-dense matrix multiplication (SpMM), and HNSW graph indexing.

---

## 🔬 Mathematical Formulations

### 1. Asymmetric Distance Computation (ADC) for Product Quantization
$$d(q, x)^2 \approx \sum_{m=1}^M \|q^m - c_{k(m)}^m\|^2$$

### 2. Compressed Sparse Row (CSR) SpMM
$$C_{ij} = \sum_{k \in \text{colIndices}(i)} A_{ik} \cdot B_{kj}$$

---

## ⚡ What's New in v5.0.0

- 📦 **`ProductQuantizationIVFPQ`**: Subspace vector decomposition into 8-bit compact codebooks and ADC table lookups.
- 📐 **`SparseDenseSpMM`**: Highly optimized CSR sparse-dense matrix multiplication kernel skipping zero floating point ops.
- 🎛️ **Studio v5.0.0**: Real-time Voronoi KMeans projection, IVF-PQ codebook explorer, and SpMM FLOP savings calculator.
- 🧪 **17/17 Tests Passing**: 100% Vitest coverage across SIMD dot products, matmul, quantization, and clustering.

---

## 🚀 Quickstart & Interactive Studio

```bash
git clone https://github.com/anderdebona/vector-tensor-wasm-engine.git
cd vector-tensor-wasm-engine
npm install
npm test
npm run build
npm start
# Open http://localhost:3009
```

---

## 📄 License & Citation
MIT License © 2026 anderdebona. See [CITATION.cff](CITATION.cff) for academic attribution.
