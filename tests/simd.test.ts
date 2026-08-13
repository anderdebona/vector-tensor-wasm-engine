import { describe, it, expect } from 'vitest';
import { SIMDVectorEngine } from '../src/simd/vector-wasm.js';
import { VectorQuantizer } from '../src/simd/quantization.js';
import { IVFFlatIndex } from '../src/simd/ann-index.js';

function randomVector(dim: number): Float32Array {
  const v = new Float32Array(dim);
  for (let i = 0; i < dim; i++) v[i] = Math.random();
  return v;
}

describe('SIMD Vector Engine', () => {
  it('should compute dot product correctly', () => {
    const a = new Float32Array([1, 2, 3, 4]);
    const b = new Float32Array([5, 6, 7, 8]);
    expect(SIMDVectorEngine.computeDotProductSIMD(a, b)).toBe(70);
  });

  it('should benchmark SIMD vs scalar', () => {
    const result = SIMDVectorEngine.benchmarkSIMDVsScalar(64, 100);
    expect(result.dimension).toBe(64);
    expect(result.speedupFactor).toBeGreaterThan(0);
  });
});

describe('Vector Quantization', () => {
  it('should scalar quantize with 4x compression', () => {
    const vec = randomVector(128);
    const result = VectorQuantizer.scalarQuantize(vec);
    expect(result.compressionRatio).toBe(4.0);
    expect(result.quantizedValues.length).toBe(128);
    expect(result.reconstructionError).toBeLessThan(0.1);
  });

  it('should dequantize back to approximate original', () => {
    const vec = new Float32Array([0.1, 0.5, 0.9, 0.3]);
    const quantized = VectorQuantizer.scalarQuantize(vec);
    const dequantized = VectorQuantizer.scalarDequantize(quantized);
    for (let i = 0; i < vec.length; i++) {
      expect(Math.abs(vec[i] - dequantized[i])).toBeLessThan(0.01);
    }
  });

  it('should product quantize into subspaces', () => {
    const vec = randomVector(128);
    const result = VectorQuantizer.productQuantize(vec, 4);
    expect(result.method).toBe('PRODUCT');
    expect(result.quantizedValues.length).toBe(128);
  });
});

describe('IVF-Flat ANN Index', () => {
  it('should build index and search for nearest neighbors', () => {
    const dim = 8;
    const index = new IVFFlatIndex(4, dim);
    const data = Array.from({ length: 50 }, (_, i) => ({
      id: `vec-${i}`,
      vector: randomVector(dim),
    }));

    index.build(data, 5);
    const query = randomVector(dim);
    const results = index.search(query, 3, 2);
    expect(results.length).toBeLessThanOrEqual(3);
    results.forEach((r) => expect(r.distance).toBeGreaterThanOrEqual(0));
  });

  it('should return index statistics', () => {
    const dim = 4;
    const index = new IVFFlatIndex(3, dim);
    const data = Array.from({ length: 20 }, (_, i) => ({
      id: `v-${i}`,
      vector: randomVector(dim),
    }));
    index.build(data, 3);
    const stats = index.getStats();
    expect(stats.numClusters).toBe(3);
    expect(stats.totalVectors).toBe(20);
  });

  it('should return results sorted by distance (ascending)', () => {
    const dim = 8;
    const index = new IVFFlatIndex(4, dim);
    const data = Array.from({ length: 100 }, (_, i) => ({
      id: `v-${i}`,
      vector: randomVector(dim),
    }));
    index.build(data, 5);
    const results = index.search(randomVector(dim), 5, 3);
    for (let i = 1; i < results.length; i++) {
      expect(results[i].distance).toBeGreaterThanOrEqual(results[i - 1].distance);
    }
  });
});

import { HNSWIndex } from '../src/simd/hnsw-index.js';
import { TensorMatMul } from '../src/simd/matmul.js';

describe('HNSW Index', () => {
  it('should insert and search vectors', () => {
    const idx = new HNSWIndex(4);
    for (let i = 0; i < 20; i++) { const v = new Float32Array(4).fill(i); idx.insert(`v${i}`, v); }
    const q = new Float32Array([5, 5, 5, 5]);
    const results = idx.search(q, 3);
    expect(results.length).toBe(3);
    expect(results[0].distance).toBeLessThanOrEqual(results[1].distance);
  });
});

describe('Tensor MatMul', () => {
  it('should multiply matrices correctly', () => {
    const a = new Float32Array([1, 2, 3, 4]); // 2x2
    const b = new Float32Array([5, 6, 7, 8]); // 2x2
    const c = TensorMatMul.multiply(a, b, 2, 2, 2);
    expect(c[0]).toBe(19); expect(c[1]).toBe(22);
  });
  it('should transpose correctly', () => {
    const m = new Float32Array([1, 2, 3, 4, 5, 6]); // 2x3
    const t = TensorMatMul.transpose(m, 2, 3);
    expect(t[0]).toBe(1); expect(t[1]).toBe(4);
  });
  it('should compute Frobenius norm', () => {
    const m = new Float32Array([3, 4]); // ||[3,4]|| = 5
    expect(TensorMatMul.frobenius(m)).toBeCloseTo(5, 5);
  });
});

describe('CosineSimilarityWasmSIMD (v4.0.0)', () => {
  it('should compute exact cosine similarity for parallel and orthogonal vectors', async () => {
    const { CosineSimilarityWasmSIMD } = await import('../src/simd/cosine-simd.js');
    const u = new Float32Array([1, 0, 0, 0]);
    const v = new Float32Array([1, 0, 0, 0]);
    expect(CosineSimilarityWasmSIMD.compute(u, v)).toBeCloseTo(1.0, 5);

    const orth = new Float32Array([0, 1, 0, 0]);
    expect(CosineSimilarityWasmSIMD.compute(u, orth)).toBeCloseTo(0.0, 5);
  });

  it('should batch rank vectors by cosine similarity', async () => {
    const { CosineSimilarityWasmSIMD } = await import('../src/simd/cosine-simd.js');
    const query = new Float32Array([1, 0, 0, 0]);
    const matrix = [
      { id: 'item-1', vector: new Float32Array([1, 0, 0, 0]) },
      { id: 'item-2', vector: new Float32Array([0.5, 0.5, 0, 0]) },
      { id: 'item-3', vector: new Float32Array([0, 1, 0, 0]) },
    ];
    const ranked = CosineSimilarityWasmSIMD.batchSearch(query, matrix, 2);
    expect(ranked.length).toBe(2);
    expect(ranked[0].id).toBe('item-1');
    expect(ranked[0].score).toBeCloseTo(1.0, 4);
  });
});

describe('KMeansClusteringEngine (v4.0.0)', () => {
  it('should cluster multi-dimensional vectors into k groups', async () => {
    const { KMeansClusteringEngine } = await import('../src/simd/kmeans-engine.js');
    const engine = new KMeansClusteringEngine(2, 10);
    const vectors = [
      new Float32Array([1.0, 1.1]),
      new Float32Array([1.2, 0.9]),
      new Float32Array([10.0, 10.1]),
      new Float32Array([9.8, 10.2]),
    ];
    const result = engine.fit(vectors);
    expect(result.centroids.length).toBe(2);
    expect(result.assignments.length).toBe(4);
    expect(result.assignments[0]).toBe(result.assignments[1]);
    expect(result.assignments[2]).toBe(result.assignments[3]);
    expect(result.assignments[0]).not.toBe(result.assignments[2]);
  });
});

