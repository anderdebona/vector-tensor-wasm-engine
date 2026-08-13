# 🗺️ Strategic Roadmap: Vector Tensor WASM Engine

Welcome to the future of browser & edge tensor computation. Accelerating high-dimensional embedding retrieval with zero runtime dependencies.

---

## 🎯 Release Milestones

### 📍 v4.0.0 — The Clustered & SIMD Cosine Era (Current)
- [x] 128-bit Float32 SIMD unrolled vector dot products.
- [x] Hierarchical Navigable Small World (HNSW) graph indexing.
- [x] Scalar Quantization (SQ8) & Product Quantization (PQ).
- [x] Tensor Matrix Multiplication & Frobenius Norms.
- [x] **CosineSimilarityWasmSIMD**: 4-wide unrolled cosine similarity batch ranker.
- [x] **KMeansClusteringEngine**: Multi-dimensional Voronoi vector partitioner.

### 📍 v4.5.0 — WebGPU WGSL Kernel Acceleration (Q4 2026)
- [ ] WebGPU WGSL compute shaders for $100\text{k}+$ batch embeddings.
- [ ] Dynamic quantization int4/fp8 support.
- [ ] Direct Arrow IPC columnar streaming.

---

## 🤝 Community Call for Contributions

We welcome issues and pull requests in the following areas:
- ⚡ AVX-512 and ARM Neon native assembly bridge.
- 📊 Benchmark comparisons with Faiss, ScaNN, and Milvus.
- 🔬 Support for hyperbolic and non-Euclidean distance metrics.
