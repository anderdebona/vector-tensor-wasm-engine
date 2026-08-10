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
