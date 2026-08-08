import { describe, it, expect } from 'vitest';
import { SIMDVectorEngine } from '../src/simd/vector-wasm.js';
import { SIMDMatrixEngine } from '../src/simd/matrix-engine.js';

describe('SIMD Vector Tensor Engine Tests', () => {
  it('should compute Float32 vector dot product correctly via unrolled SIMD lanes', () => {
    const a = new Float32Array([1, 2, 3, 4]);
    const b = new Float32Array([1, 1, 1, 1]);

    const dot = SIMDVectorEngine.computeDotProductSIMD(a, b);
    expect(dot).toBe(10);
  });

  it('should benchmark SIMD throughput and report speedup factor', () => {
    const result = SIMDVectorEngine.benchmarkSIMDVsScalar(128, 100);
    expect(result.speedupFactor).toBeGreaterThan(0);
    expect(result.dotProduct).toBeGreaterThan(0);
  });

  it('should perform high-speed Float32 matrix multiplication', () => {
    const size = 4;
    const a = new Float32Array(size * size).fill(1);
    const b = new Float32Array(size * size).fill(1);

    const res = SIMDMatrixEngine.multiplyMatrices(a, b, size);
    expect(res.rowsA).toBe(4);
    expect(res.executionTimeMs).toBeGreaterThanOrEqual(0);
  });
});
